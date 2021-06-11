import train
import render
import convnet_ai


# ai = convnet_ai.RotatedCenteredAI(train.train_settings, "models_output/centered-rotated-ai-2020-02-29 12:29-last.h5")
ai = convnet_ai.RotatedCenteredAI(train.train_settings, "./models_output/centered-rotated-ai-2021-06-10 09:29:47-last.h5")
ai.epsilon = 0.001
for _ in range(3):
    renderer = render.Renderer(ai)
    renderer.render_loop()
