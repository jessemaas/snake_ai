import render
import convnet_ai

ai = convnet_ai.RotatedCenteredAI("models/rotated_centered_conv_decreasing_epsilon_and_lr-last.h5")
ai.epsilon = 0.01
for _ in range(3):
    renderer = render.Renderer(ai)
    renderer.render_loop()