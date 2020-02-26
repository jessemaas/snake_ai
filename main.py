import render
import convnet_ai

ai = convnet_ai.CenteredAI("models_output/centered-ai-2020-02-26 11:27-last.h5")
for _ in range(3):
    renderer = render.Renderer(ai)
    renderer.render_loop()