/// Configuration parameters for Flow Matching
pub struct CfmParams {
    /// Minimum sigma for flow matching
    pub sigma_min: f64,
    /// Solver type
    pub solver: String,
    /// Time scheduler type
    pub t_scheduler: String,
    /// Classifier-free guidance rate for training
    pub training_cfg_rate: f64,
    /// Classifier-free guidance rate for inference
    pub inference_cfg_rate: f64,
    /// Regularization loss type
    pub reg_loss_type: String,
}

impl Default for CfmParams {
    fn default() -> Self {
        Self {
            sigma_min: 1e-06,
            solver: "euler".to_string(),
            t_scheduler: "cosine".to_string(),
            training_cfg_rate: 0.2,
            inference_cfg_rate: 0.7,
            reg_loss_type: "l1".to_string(),
        }
    }
}
