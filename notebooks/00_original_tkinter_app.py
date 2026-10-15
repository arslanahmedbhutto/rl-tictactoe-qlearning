import tkinter as tk
from tkinter import messagebox
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
import os
import threading

# ======================================
# ENVIRONMENT
# ======================================

class TicTacToe:

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros(9)
        return tuple(self.board)

    def available_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def check_winner(self):
        wins = [(0,1,2),(3,4,5),(6,7,8),
                (0,3,6),(1,4,7),(2,5,8),
                (0,4,8),(2,4,6)]

        for a,b,c in wins:
            if self.board[a] == self.board[b] == self.board[c] != 0:
                return self.board[a]
        return None


# ======================================
# Q LEARNING AGENT
# ======================================

class QAgent:

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.q_table = defaultdict(lambda: np.zeros(9))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)

        q_vals = self.q_table[state]
        return max(actions, key=lambda x: q_vals[x])

    def update(self, state, action, reward, next_state, done):
        max_next = 0 if done else np.max(self.q_table[next_state])

        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * max_next - self.q_table[state][action]
        )

    def save(self):
        with open("q_model.pkl", "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self):
        if os.path.exists("q_model.pkl"):
            with open("q_model.pkl", "rb") as f:
                data = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(9), data)


# ======================================
# GUI
# ======================================

class TicTacToeGUI:

    def __init__(self, root):

        self.root = root
        self.root.title("Professional RL Tic-Tac-Toe")

        self.env = TicTacToe()
        self.agent = QAgent()

        self.show_values = True

        self.buttons = []
        self.create_board()

        self.status = tk.Label(root, text="Ready",
                               font=("Arial", 12),
                               bg="black", fg="white")
        self.status.grid(row=3, column=0, columnspan=3, sticky="we")

        # Controls
        tk.Button(root, text="Train AI",
                  command=self.start_training,
                  bg="orange").grid(row=4, column=0, sticky="we")

        tk.Button(root, text="Student Mode",
                  command=self.toggle_student).grid(row=4, column=1, sticky="we")

        tk.Button(root, text="Evaluate",
                  command=self.evaluate_agent).grid(row=4, column=2, sticky="we")

        tk.Button(root, text="Save",
                  command=self.agent.save).grid(row=5, column=0, sticky="we")

        tk.Button(root, text="Load",
                  command=self.load_model).grid(row=5, column=1, sticky="we")

        tk.Button(root, text="Reset",
                  command=self.reset_game).grid(row=5, column=2, sticky="we")


    # ======================================
    # BOARD
    # ======================================

    def create_board(self):
        for i in range(9):
            btn = tk.Button(self.root,
                            text="",
                            font=("Arial", 20),
                            width=6,
                            height=3,
                            command=lambda i=i: self.human_move(i))
            btn.grid(row=i//3, column=i%3)
            self.buttons.append(btn)


    # ======================================
    # GAMEPLAY
    # ======================================

    def human_move(self, i):

        if self.env.board[i] != 0:
            return

        self.env.board[i] = -1
        self.update_board()

        if self.env.check_winner() == -1:
            self.end_game("You Win!")
            return

        if len(self.env.available_actions()) == 0:
            self.end_game("Draw!")
            return

        self.root.after(200, self.ai_move)

    def ai_move(self):

        state = tuple(self.env.board)
        actions = self.env.available_actions()
        q_vals = self.agent.q_table[state]

        action = self.agent.choose_action(state, actions)

        self.status.config(
            text=f"AI Move: {action} | Q-value={q_vals[action]:.3f}"
        )

        self.env.board[action] = 1
        self.update_board()

        if self.env.check_winner() == 1:
            self.end_game("AI Wins!")
        elif len(self.env.available_actions()) == 0:
            self.end_game("Draw!")

    def update_board(self):

        state = tuple(self.env.board)
        q_vals = self.agent.q_table[state]
        actions = self.env.available_actions()

        best_val = max([q_vals[a] for a in actions]) if actions else 0

        for i in range(9):

            if self.env.board[i] == 1:
                self.buttons[i].config(text="X", bg="white", fg="red")

            elif self.env.board[i] == -1:
                self.buttons[i].config(text="O", bg="white", fg="blue")

            else:

                # Heatmap coloring
                if actions:
                    val = q_vals[i]
                    norm = (val + 1) / 2
                    green = int(255 * norm)
                    color = f'#%02x%02x%02x' % (255-green, 255, 255-green)
                else:
                    color = "lightyellow"

                arrow = " ↑" if i in actions and q_vals[i] == best_val else ""

                if self.show_values:
                    self.buttons[i].config(
                        text=f"{q_vals[i]:.2f}{arrow}",
                        bg=color
                    )
                else:
                    self.buttons[i].config(text="", bg="lightyellow")

    def reset_game(self):
        self.env.reset()
        self.update_board()

    def end_game(self, msg):
        messagebox.showinfo("Game Over", msg)
        self.reset_game()


    # ======================================
    # TRAINING (Thread-Safe)
    # ======================================

    def start_training(self):
        threading.Thread(target=self.train_ai).start()

    def train_ai(self):

        episodes = 20000
        wins = losses = draws = 0

        win_curve = []
        loss_curve = []
        draw_curve = []

        for ep in range(episodes):

            self.env.reset()
            state = tuple(self.env.board)
            done = False

            while not done:

                action = self.agent.choose_action(state, self.env.available_actions())
                self.env.board[action] = 1

                winner = self.env.check_winner()
                if winner == 1:
                    self.agent.update(state, action, 1, state, True)
                    wins += 1
                    break

                if len(self.env.available_actions()) == 0:
                    self.agent.update(state, action, 0, state, True)
                    draws += 1
                    break

                opponent = random.choice(self.env.available_actions())
                self.env.board[opponent] = -1

                next_state = tuple(self.env.board)
                winner = self.env.check_winner()

                if winner == -1:
                    reward = -1
                    losses += 1
                    done = True
                elif len(self.env.available_actions()) == 0:
                    reward = 0
                    draws += 1
                    done = True
                else:
                    reward = 0

                self.agent.update(state, action, reward, next_state, done)
                state = next_state

            win_curve.append(wins)
            loss_curve.append(losses)
            draw_curve.append(draws)

        self.agent.epsilon = 0

        # Safe plotting in main thread
        self.root.after(0, lambda:
                        self.finish_training_plot(win_curve, loss_curve, draw_curve))

    def finish_training_plot(self, wins, losses, draws):

        self.status.config(text="Training Complete!")

        plt.figure()
        plt.plot(wins, label="Wins")
        plt.plot(losses, label="Losses")
        plt.plot(draws, label="Draws")
        plt.legend()
        plt.title("Training Performance")
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Count")
        plt.show()

        self.reset_game()


    # ======================================
    # EVALUATION
    # ======================================

    def evaluate_agent(self):

        games = 1000
        win = loss = draw = 0

        for _ in range(games):

            self.env.reset()
            done = False

            while not done:

                state = tuple(self.env.board)
                action = self.agent.choose_action(state,
                                                  self.env.available_actions())
                self.env.board[action] = 1

                winner = self.env.check_winner()
                if winner == 1:
                    win += 1; break

                if len(self.env.available_actions()) == 0:
                    draw += 1; break

                opponent = random.choice(self.env.available_actions())
                self.env.board[opponent] = -1

                winner = self.env.check_winner()
                if winner == -1:
                    loss += 1; break

                if len(self.env.available_actions()) == 0:
                    draw += 1; break

        rate = win / games * 100

        messagebox.showinfo("Evaluation Results",
                            f"Win: {win}\nLoss: {loss}\nDraw: {draw}\nWin Rate: {rate:.2f}%")

    # ======================================
    # TOGGLES
    # ======================================

    def toggle_student(self):
        self.show_values = not self.show_values
        mode = "Student Mode ON" if not self.show_values else "Instructor Mode"
        self.status.config(text=mode)
        self.update_board()

    def load_model(self):
        self.agent.load()
        self.agent.epsilon = 0
        self.status.config(text="Model Loaded")
        self.update_board()


# ======================================
# RUN
# ======================================

root = tk.Tk()
app = TicTacToeGUI(root)
root.mainloop()
