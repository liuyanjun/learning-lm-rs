use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    /// 隐藏状态维度
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    /// mlp 中间状态维度
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }
    /// inference unit
    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        /// 输入文本长度 6
        let seq_len = input.size();
        /// 已处理文本长度
        let past_seq_len = cache.len();

        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        /// 计算 q 与 kv 的多头关系（倍数）
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        /// shape = 6 * 128
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        /// shape = 6 * 128
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        /// 6 * 4 * 16
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        /// shape = 4 * 2 * 6 * 6
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        /// shape = 6 * 384
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        /// 获取文本的词向量
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            /// 归一化
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            /// 计算 q, k, v projection
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            /// 对q, k 施加位置嵌入旋转变换 
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            //todo!("self_attention(...)");
            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv
            );
            hidden_states.print();
            //todo!("down_proj matmul and add residual");
            OP::matmul_transb(&mut residual, 1f32, &hidden_states, &self.params.wo[layer], 1.0f32);
            //todo!("mlp(...)");
            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps

            )
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32>{
        let mut result = Vec::<u32>::new();
        
        //todo!("实现文本生成");
        for token in token_ids {
            result.push(*token);
        }
        let mut input = Tensor::new(Vec::from(token_ids), &vec![1, token_ids.len()]);
        let mut cache = self.new_cache();
        for _ in 0..max_len {
            let embed = self.forward(&input, &mut cache);
            let token = OP::random_sample(&embed, top_p, top_k, temperature);
            if token == 2{
                break;
            }
            result.push(token);
            input = Tensor::new(vec![token], &vec![1, 1]);
        }
        result
    }
}

fn self_attention(
    
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    //todo!("Implement self_attention");
    
        // x = rms_norm(residual)
        // Q = RoPE(x @ Q_weight.T)
        // K = RoPE(x @ K_weight.T)
        // V = x @ V_weight.T
        // K = cat(K_cache, K)
        // V = cat(V_cache, V)
        // ### 以下是你需要实现的部分

        // score = Q @ K.T / sqrt(dim)
        // q.reshape(&vec![seq_len, ])
        // attn = softmax(score)
        // x = attn @ V
        // x = x @ O_weight.T
        // residual = x + residual
    let qkv_dim = dqkv;
    let seq_dim = n_kv_h * dqkv;
    let hidden_len = n_kv_h * n_groups * dqkv;
    let hidden_data = unsafe{
        hidden_states.data_mut()
    };
    let att_dim_3 = total_seq_len;
    let att_dim_2 = seq_len * total_seq_len;
    let att_dim_1 = n_groups * seq_len * total_seq_len;
    let att_ptr = unsafe{
        att_scores.data_mut()
    };
    for x in 0..seq_len {
        for y in 0..total_seq_len {
            for i in 0..n_kv_h {
                for group in 0..n_groups {
                    let start_q = (i *n_groups + group) * qkv_dim + seq_dim * n_groups * x;
                    let q_vec = &q.slice(start_q, &vec![16, 1]);
                    let start_k = i * qkv_dim + seq_dim * y;
                    let k_vec = &k.slice(start_k, &vec![16, 1]);
                    let value = OP::dot(q_vec, k_vec)/f32::sqrt(qkv_dim as f32);
                    
                    att_ptr[i * att_dim_1 + group * att_dim_2 + x * att_dim_3 + y] = value;
                }
            }
        }
    }

    OP::masked_softmax(att_scores);
    let v_ptr = v.data();
    for i in 0..n_kv_h {
        for g in 0..n_groups {
            let att_start = att_dim_1 * i + g * att_dim_2;
            let att_mat = &att_scores.slice(att_start, &vec![seq_len, total_seq_len]);
            let mut data = vec![0f32; dqkv * total_seq_len];
            for row in 0..dqkv{
                let d_start = row * total_seq_len;
                for col in 0..total_seq_len{
                    data[d_start + col] = v_ptr[col * dqkv * n_kv_h + i*dqkv + row];
                }
            }
            let v_mat = Tensor::new(data, &vec![dqkv, total_seq_len]);
            let mut t_mat = Tensor::default(&vec![seq_len, dqkv]);
            OP::matmul_transb(&mut t_mat, 0f32, att_mat, &v_mat, 1f32);
            let t_data = t_mat.data();
            for row in 0..seq_len{
                for col in 0..dqkv{
                    let hidden_p = row * hidden_len + (i * n_groups + g) *dqkv + col;
                    hidden_data[hidden_p] = t_data[row * dqkv + col];
                }
            }
        }
    }
}

fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    //todo!("Implement mlp");
    /*
    itermediate = gate * sigmoid(gate) * up ## silu
    output = itermediate @ down_weight.T
    residual = output + residual
    */

    // hidden = rms_norm(residual)
    OP::rms_norm( hidden_states, residual, rms_w, eps);
    // gate = hidden @ gate_weight.T
    OP::matmul_transb(gate, 0., hidden_states, w_gate, 1.);
    // up = hidden @ up_weight.T
    OP::matmul_transb(up, 0., hidden_states, w_up, 1.);
    // itermediate = gate * sigmoid(gate) * up ## silu

    OP::silu(up, gate);
    // output = itermediate @ down_weight.T
    OP::matmul_transb(hidden_states, 0., up, w_down, 1.);
    // residual = output + residual
    OP::add(residual, &hidden_states);

}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}

// pub fn test_ld_safetensors() {
//     use std::path::PathBuf;
//     use crate::tensor::float_eq;
//     let project_dir = env!("CARGO_MANIFEST_DIR");
//     let model_dir = PathBuf::from(project_dir).join("models").join("story");
//     //let model = Llama::from_safetensors(model_dir);

//     let model_file = std::fs::read(model_dir.join("model.safetensors")).unwrap();
//     let safetensor = SafeTensors::deserialize(&model_file).unwrap();
//     println!("{:?}", safetensor.names());
    
//     // assert_eq!(loaded.names(), vec!["test"]);
//         // let tensor = loaded.tensor("test").unwrap();
//         // assert_eq!(tensor.shape(), vec![2, 2]);
//         // assert_eq!(tensor.dtype(), Dtype::I32);
        
// /* 
//         ["lm_head.weight", 
//         "model.layers.0.post_attention_layernorm.weight", 
//         "model.layers.1.self_attn.o_proj.weight", 
//         "model.layers.1.input_layernorm.weight", 
//         "model.layers.0.mlp.up_proj.weight", 
//         "model.layers.0.self_attn.v_proj.weight", 
//         "model.layers.1.self_attn.v_proj.weight", 
//         "model.layers.0.self_attn.o_proj.weight", 
//         "model.layers.1.self_attn.q_proj.weight", 
//         "model.layers.0.input_layernorm.weight", 
//         "model.layers.1.mlp.up_proj.weight", 
//         "model.layers.1.mlp.gate_proj.weight", 
//         "model.layers.1.self_attn.k_proj.weight", 
//         "model.norm.weight", 
//         "model.layers.1.mlp.down_proj.weight", 
//         "model.layers.0.self_attn.q_proj.weight", 
//         "model.layers.0.self_attn.k_proj.weight", 
//         "model.layers.0.mlp.down_proj.weight", 
//         "model.layers.1.post_attention_layernorm.weight", 
//         "model.layers.0.mlp.gate_proj.weight"]
//         */

// }
