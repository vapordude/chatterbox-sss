pub struct CfmParams {
    pub sigma_min: f64,
    pub solver: &'static str,
    pub t_scheduler: &'static str,
    pub training_cfg_rate: f64,
    pub inference_cfg_rate: f64,
    pub reg_loss_type: &'static str,
}

pub const CFM_PARAMS: CfmParams = CfmParams {
    sigma_min: 1e-06,
    solver: "euler",
    t_scheduler: "cosine",
    training_cfg_rate: 0.2,
    inference_cfg_rate: 0.7,
    reg_loss_type: "l1",
};
