Model loading failed: Error(s) in loading state_dict for SelfAttention:
        Missing key(s) in state_dict: "output_scale". 
Traceback (most recent call last):
  File "/home/user/yaojie/Anomagic-main/demo/app.py", line 268, in load_generator
    generator.load_models()
  File "/home/user/yaojie/Anomagic-main/demo/app.py", line 141, in load_models
    self.anomagic_model = Anomagic(self.pipe, self.clip_vision_model, self.ip_ckpt_path, self.att_ckpt_path,
  File "/home/user/yaojie/Anomagic-main/demo/ip_adapter/ip_adapter_anomagic.py", line 166, in __init__
    self.load_anomagic()
  File "/home/user/yaojie/Anomagic-main/demo/ip_adapter/ip_adapter_anomagic.py", line 273, in load_anomagic
    self.attention_module.load_state_dict(att_state_dict, strict=True)
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/torch/nn/modules/module.py", line 2581, in load_state_dict
    raise RuntimeError(
RuntimeError: Error(s) in loading state_dict for SelfAttention:
        Missing key(s) in state_dict: "output_scale".
