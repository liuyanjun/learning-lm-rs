use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::{SafeTensors, View};
use std::slice;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        //todo!("实现从safetensors文件的模型参数加载");
        let layers = config.num_hidden_layers;
        println!("{:?}", safetensor.names());

        let get_tensor = |name: &str| {
            match safetensor.tensor(name) {
                Ok(input) => {
                    let len = input.shape().iter().product();
                    let data_slice = unsafe{slice::from_raw_parts(input.data().as_ptr() as *const f32, len)};
                    Tensor::new(Vec::from(data_slice), &input.shape().to_vec())
                },
                Err(e) => {Tensor::default(&Vec::new())},
            }
         };
         /* 
        ["lm_head.weight", 
        "model.layers.0.post_attention_layernorm.weight", 
        "model.layers.1.self_attn.o_proj.weight", 
        "model.layers.1.input_layernorm.weight", 
        "model.layers.0.mlp.up_proj.weight", 
        "model.layers.0.self_attn.v_proj.weight", 
        "model.layers.1.self_attn.v_proj.weight", 
        "model.layers.0.self_attn.o_proj.weight", 
        "model.layers.1.self_attn.q_proj.weight", 
        "model.layers.0.input_layernorm.weight", 
        "model.layers.1.mlp.up_proj.weight", 
        "model.layers.1.mlp.gate_proj.weight", 
        "model.layers.1.self_attn.k_proj.weight", 
        "model.norm.weight", 
        "model.layers.1.mlp.down_proj.weight", 
        "model.layers.0.self_attn.q_proj.weight", 
        "model.layers.0.self_attn.k_proj.weight", 
        "model.layers.0.mlp.down_proj.weight", 
        "model.layers.1.post_attention_layernorm.weight", 
        "model.layers.0.mlp.gate_proj.weight"]
        */
        
        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            // decoder layer
            // pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
            rms_att_w: (0..layers).map(|i| get_tensor(&format!("model.layers.{i}.input_layernorm.weight"))).collect(),
            // pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
            wq: (0..layers).map(|i| get_tensor(&format!("model.layers.{i}.self_attn.q_proj.weight"))).collect(),
            // pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
            wk: (0..layers).map(|i| get_tensor(&format!("model.layers.{i}.self_attn.k_proj.weight"))).collect(),
            // pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
            wv: (0..layers).map(|i| get_tensor(&format!("model.layers.{i}.self_attn.v_proj.weight"))).collect(),
            // pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
            wo: (0..layers).map(|i| get_tensor(&format!("model.layers.{i}.self_attn.o_proj.weight"))).collect(),
            // // ffn layer
            // pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
            rms_ffn_w: (0..layers).map(|i| get_tensor(&format!("model.layers.{i}.post_attention_layernorm.weight"))).collect(),
            // pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
            w_up: (0..layers).map(|i| get_tensor(&format!("model.layers.{i}.mlp.up_proj.weight"))).collect(),
            // pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
            w_gate: (0..layers).map(|i| get_tensor(&format!("model.layers.{i}.mlp.gate_proj.weight"))).collect(),
            // pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
            w_down: (0..layers).map(|i| get_tensor(&format!("model.layers.{i}.mlp.down_proj.weight"))).collect(),
            // // output
            // pub rms_out_w: Tensor<T>, // (hidden_size, )
            rms_out_w: get_tensor("model.norm.weight"),
            // pub lm_head: Tensor<T>,   // (vocab_size, dim)
            lm_head: get_tensor("lm_head.weight"),
            
        }

        // assert_eq!(loaded.names(), vec!["test"]);
        // let tensor = loaded.tensor("test").unwrap();
        // assert_eq!(tensor.shape(), vec![2, 2]);
        // assert_eq!(tensor.dtype(), Dtype::I32);
        // // 16 bytes
        // assert_eq!(tensor.data(), b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0");
    }
}
