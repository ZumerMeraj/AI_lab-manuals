import gymnasium as gym
import pygame

# Create environment
env = gym.make("MountainCar-v0", render_mode="human")

font = None
scores = []

NUM_EPISODES = 50  # Task 1: 50 episodes

for episode in range(1, NUM_EPISODES + 1):
    state, info = env.reset()
    done = False
    score = 0
    steps = 0

    while not done:
        # Task 2: Print state variables
        print("State:", state)  # [position, velocity]

        # Task 7: Rule-based action
        if state[1] > 0:
            action = 2  # push right
        else:
            action = 0  # push left

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward
        steps += 1

        # Initialize font for Pygame
        if font is None:
            pygame.font.init()
            font = pygame.font.SysFont("Arial", 24)

        # Task 3: Display Episode + Score in blue
        surface = pygame.display.get_surface()
        text = font.render(f"Episode: {episode} | Score: {int(score)}", True, (0, 0, 255))
        surface.blit(text, (200, 20))

        # Task 5: Slow down visualization
        pygame.time.delay(20)
        pygame.display.update()

    scores.append(score)
    print(f"Episode {episode} Steps: {steps} Score: {score}")

# Task 4: Best score
print("\nBest Score Achieved:", max(scores))

env.close()
pygame.quit()
