import train
import render
import convnet_ai


# ai = convnet_ai.RotatedCenteredAI("models_output/centered-rotated-ai-2020-02-29 12:29-last.h5")
ai = convnet_ai.RotatedCenteredAI("models_output/centered-rotated-ai-2020-03-20 09:52:24-above-12.h5")
ai.epsilon = 0.0
for _ in range(3):
    renderer = render.Renderer(ai)
    renderer.render_loop()