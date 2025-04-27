from math import sqrt

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class TimeSeriesTokenization(nn.Module):
    def __init__(self, input_dim, codebook_size, codebook_dim):
        super().__init__()
        self.codebook_size = codebook_size
        # 可学习的码本 (每个token对应一个向量)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        # 分箱边界参数 (可学习)
        self.bin_bounds = nn.Parameter(torch.linspace(0, 1, codebook_size - 1))

    def forward(self, x):
        """时间序列标记化
        Args:
            x: input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            quantized: 量化后的嵌入 (batch_size, seq_len, codebook_dim)
        """
        # 1. 平均缩放 (Mean Scaling)
        mean = x.mean(dim=1, keepdim=True)
        scaled_x = x / (mean + 1e-6)  # 防止除零

        # 2. 分箱量化 (Binning Quantization)
        # 将值映射到[0,1]区间
        min_val = scaled_x.min(dim=1, keepdim=True)[0]
        max_val = scaled_x.max(dim=1, keepdim=True)[0]
        normalized = (scaled_x - min_val) / (max_val - min_val + 1e-6)

        # 计算分箱索引 (通过可学习边界)
        bins = torch.cat([-torch.inf, self.bin_bounds, torch.inf]).unsqueeze(0).unsqueeze(0)  # (1,1,codebook_size+1)
        indices = torch.bucketize(normalized, bins) - 1  # (batch_size, seq_len, input_dim)

        # 3. 码本查找
        quantized = self.codebook(indices)  # (batch_size, seq_len, input_dim, codebook_dim)
        quantized = quantized.mean(dim=2)  # 沿特征维度平均

        # 直通估计器 (保持梯度流)
        return scaled_x + (quantized - scaled_x).detach()


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        # 基础配置参数
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout

        # LLM相关参数
        self.d_llm = configs.llm_dim
        self.llm_layers = configs.llm_layers
        self.llm_model_name = configs.llm_model

        # 时间序列标记化参数
        self.codebook_size = getattr(configs, 'codebook_size', 256)  # 默认256个分箱
        self.use_residual = getattr(configs, 'token_residual', True)  # 是否使用残差连接

        # 加载预训练LLM
        self._init_llm_model(configs)

        # 输入特征处理
        self.patch_embedding = PatchEmbedding(
            configs.d_model,
            patch_len=configs.patch_len,
            stride=configs.stride,
            dropout=configs.dropout
        )

        # 时间序列标记化层
        self.tokenization_layer = TimeSeriesTokenization(
            input_dim=self.d_model,
            codebook_size=self.codebook_size,
            codebook_dim=self.d_llm
        )

        # 维度适配层
        if self.d_model != self.d_llm:
            self.adapter = nn.Sequential(
                nn.Linear(self.d_model, self.d_llm),
                nn.GELU(),
                nn.LayerNorm(self.d_llm)
            )
        else:
            self.adapter = nn.Identity()

        # 输出层结构
        self.patch_nums = int((configs.seq_len - configs.patch_len) / configs.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums
        self.output_projection = FlattenHead(
            n_vars=configs.enc_in,
            nf=self.head_nf,
            target_window=configs.pred_len,
            head_dropout=configs.dropout
        )

        # 归一化层
        self.normalize_layers = Normalize(configs.enc_in, affine=False)

                # 添加prompt的可学习编码层
        self.prompt_projection = nn.Linear(self.d_llm, self.d_llm)
        self.prompt_dropout = nn.Dropout(configs.dropout)

        # 残差连接参数
        if self.use_residual:
            self.res_weight = nn.Parameter(torch.tensor(0.1))  # 可学习的残差权重

    def _init_llm_model(self, configs):
        """初始化LLM主干网络"""
        if configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llm_model = LlamaModel.from_pretrained('huggyllama/llama-7b', config=self.llama_config)
            self.tokenizer = LlamaTokenizer.from_pretrained('huggyllama/llama-7b')

        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('gpt2')
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.llm_model = GPT2Model.from_pretrained('gpt2', config=self.gpt2_config)
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        elif configs.llm_model == 'deepseek':
            model_name = "deepseek-ai/deepseek-llm-7b-base"
            self.llm_model = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 冻结LLM参数
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # 处理特殊token
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else '[PAD]'

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        完整前向传播流程
        Args:
            x_enc: 编码器输入 (batch_size, seq_len, n_vars)
            x_mark_enc: 编码器时间标记 (batch_size, seq_len, n_time_features)
            x_dec: 解码器输入 (batch_size, label_len + pred_len, n_vars)
            x_mark_dec: 解码器时间标记 (batch_size, label_len + pred_len, n_time_features)
        Returns:
            output: 预测结果 (batch_size, pred_len, n_vars)
        """
        # 任务路由
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # 完整预测流程
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

            # 截取预测区间并调整维度
            return dec_out[:, -self.pred_len:, :]  # (batch_size, pred_len, n_vars)

        elif self.task_name == 'classification':
            # 分类任务流程
            cls_emb = self._extract_global_features(x_enc)
            return self.classifier(cls_emb)

        elif self.task_name == 'anomaly_detection':
            # 异常检测流程
            recon = self.reconstruct(x_enc)
            return torch.abs(x_enc - recon)  # 重构误差作为异常分数

        else:
            raise NotImplementedError(f"未实现的任务类型: {self.task_name}")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 归一化处理
        x_enc = self.normalize_layers(x_enc, 'norm')  # (batch_size, seq_len, n_vars)

        # 维度预处理
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous()  # (batch_size, n_vars, seq_len)

        # 计算统计特征
        patch_data = x_enc.reshape(B * N, T)  # (batch_size*n_vars, seq_len)
        min_vals = patch_data.min(dim=1)[0]  # (batch_size*n_vars,)
        max_vals = patch_data.max(dim=1)[0]  # (batch_size*n_vars,)
        medians = patch_data.median(dim=1).values  # (batch_size*n_vars,)
        trends = patch_data[:, -1] - patch_data[:, 0]  # (batch_size*n_vars,)
        lags = self.calculate_lags(patch_data)  # (batch_size*n_vars, top_k)

        # 生成统计描述prompt
        prompts = []
        for i in range(B * N):
            prompt = (
                f"<|start_prompt|>Dataset: {self.description} "
                f"Statistics: min={min_vals[i]:.2f}, max={max_vals[i]:.2f}, "
                f"median={medians[i]:.2f}, trend={'up' if trends[i] > 0 else 'down'}, "
                f"top_lags={lags[i].tolist()}<|<end_prompt>|>"
            )
            prompts.append(prompt)

        # 生成prompt嵌入
        prompt_ids = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).input_ids.to(x_enc.device)
        prompt_embeds = self.llm_model.get_input_embeddings()(prompt_ids)  # (B*N, p_seq, d_llm)

        # 时间序列编码
        enc_out = self.patch_embedding(x_enc)  # (batch_size, n_vars, n_patches, d_model)
        enc_out = enc_out.permute(0, 2, 1, 3)  # (batch_size, n_patches, n_vars, d_model)
        enc_out = enc_out.reshape(B * N, -1, self.d_model)  # (B*N, n_patches, d_model)

        # 维度适配与标记化
        enc_out = self.adapter(enc_out)  # (B*N, n_patches, d_llm)
        ts_embeds = self.tokenization_layer(enc_out)  # (B*N, n_patches, d_llm)

        # 拼接提示与序列
        combined_embeds = torch.cat([
            prompt_embeds,
            ts_embeds
        ], dim=1)  # (B*N, p_seq+n_patches, d_llm)

        # LLM处理
        llm_outputs = self.llm_model(
            inputs_embeds=combined_embeds,
            output_hidden_states=True
        )
        last_hidden = llm_outputs.last_hidden_state  # (B*N, total_seq, d_llm)

        # 提取时序相关特征
        ts_features = last_hidden[:, prompt_embeds.size(1):, :]  # (B*N, n_patches, d_llm)
        ts_features = ts_features.reshape(B, N, -1, self.d_llm)  # (B, N, n_patches, d_llm)

        # 输出投影
        dec_out = self.output_projection(ts_features)  # (B, N, pred_len)
        dec_out = dec_out.permute(0, 2, 1)  # (B, pred_len, N)

        # 反归一化
        dec_out = self.normalize_layers(dec_out, 'denorm')
        return dec_out  # (batch_size, pred_len, n_vars)

    def calculate_lags(self, x):
        """计算时间序列的显著滞后项"""
        # x: (batch_size*n_vars, seq_len)
        autocorr = torch.stack([
            torch.fft.irfft(
                torch.fft.rfft(x) * torch.conj(torch.fft.rfft(x)),
                n=x.size(-1)
            ) for _ in range(self.top_k)
        ], dim=-1)
        _, indices = torch.topk(autocorr.mean(dim=0), self.top_k, dim=-1)
        return indices