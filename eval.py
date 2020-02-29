import train
import convnet_ai

ai = convnet_ai.RotatedCenteredAI("models/RotatedCenteredAI_no_moving_backwards-last.h5")

for _ in range(10):
    trainer = train.Trainer(ai, 512)
    trainer.simulate_entire_game()
    max_score, total_score = trainer.results()
    print("max_score:", max_score, "avg_score:", total_score / 512)
