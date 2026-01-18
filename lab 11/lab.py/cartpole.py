import gymnasium as gym
import pygame

# Create environment
env = gym.make("CartPole-v1", render_mode="human")

font = None
scores = []

# Task 1: Run 50 episodes
for episode in range(1, 51):
    score = 0
    state, info = env.reset()
    done = False

    while not done:
        # Task 8: Rule-based action based on pole angle
        if state[2] > 0:
            action = 1  # push right
        else:
            action = 0  # push left

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward

        # Initialize font for Pygame
        if font is None:
            pygame.font.init()
            font = pygame.font.SysFont("Arial", 24)

        # Get surface and display Episode + Score (Task 2 & 3)
        surface = pygame.display.get_surface()
        text = font.render(f"Episode: {episode} | Score: {int(score)}", True, (0, 255, 0))
        surface.blit(text, (200, 20))

        # Task 5: slow down
        pygame.time.delay(20)
        pygame.display.update()

    scores.append(score)
    print(f"Episode {episode} Score: {score}")

# Task 4: Maximum score
print("\nMaximum Score Achieved:", max(scores))

env.close()
pygame.quit()
