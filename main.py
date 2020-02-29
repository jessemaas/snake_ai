import render
import convnet_ai

# ai = convnet_ai.RotatedCenteredAI("models_output/centered-rotated-ai-2020-02-29 12:29-last.h5")
ai = convnet_ai.RotatedCenteredAI("models/reducing_epsilon_and_lr_RotatedCenteredAI-last.h5")
ai.epsilon = 0.01
for _ in range(3):
    renderer = render.Renderer(ai)
    renderer.render_loop()