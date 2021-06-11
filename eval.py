import train
import convnet_ai
import ai as ai_module

ai = convnet_ai.RotatedCenteredAI(train.train_settings, "./models_output/centered-rotated-ai-2021-06-10 09:29:47-last.h5")
ai.epsilon = 0.001

for _ in range(10):
    trainer = train.Trainer(ai, 512)
    trainer.simulate_entire_game()
    max_score, total_score = trainer.results()
    print("max_score:", max_score, "avg_score:", total_score / 512)
