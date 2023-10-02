# Implementation of AdjointDPM: Adjoint Sensitivity Method for Gradient Backpropagation of Diffusion Probabilistic Models

AdjointDPM is a method that can optimize the parameters of DPMs, including network parameters, text embedding and noise states,  when the objective is a differentiable metric defined on the generated contents. There are several interesting experiments to demonstrate the effectiveness of AdjointDPM. For the tasks: vocabulary expansion, stylization, security auditing under an NSFW filter, and text embedding inversion, they are implemented based on [🧨 Diffusers](https://github.com/huggingface/diffusers). Check them in diffuser/examples. For security auditing under an ImageNet classifier, we implement the code heavily based on [dpm-solver](https://github.com/LuChengTHU/dpm-solver/tree/main/examples/ddpm_and_guided-diffusion) codebase.


