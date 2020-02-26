import render
import convnet_ai

ai = convnet_ai.CenteredAI("models_output/centered-ai-2020-02-25 21:34:29.867836.h5")
renderer = render.Renderer(ai)
renderer.render_loop()