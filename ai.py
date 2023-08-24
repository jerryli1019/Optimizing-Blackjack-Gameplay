import copy
import random

from game import Game, states

HIT = 0
STAND = 1
DISCOUNT = 0.95 #This is the gamma value for all value calculations

class Agent:
    def __init__(self):

        self.MC_values = {} 
        self.S_MC = {}      
        self.N_MC = {}      

        self.TD_values = {}  
        self.N_TD = {} 

        self.Q_values = {} 
        self.N_Q = {}   

        for s in states:
            self.MC_values[s] = 0
            self.S_MC[s] = 0
            self.N_MC[s] = 0
            self.TD_values[s] = 0
            self.N_TD[s] = 0
            self.Q_values[s] = [0,0] 
            self.N_Q[s] = [0,0] 
  
        self.simulator = Game()

    @staticmethod
    def default_policy(state):
        user_sum = state[0]
        user_A_active = state[1]
        actual_user_sum = user_sum + user_A_active * 10
        if actual_user_sum < 14:
            return 0
        else:
            return 1

    @staticmethod
    def alpha(n):
        return 10.0/(9 + n)
   
    def make_one_transition(self, action):
        if self.simulator.game_over():
            return None
        if action == HIT:
            self.simulator.act_hit()
        elif action == STAND:
            self.simulator.act_stand()
        return self.simulator.state

    def MC_run(self, num_simulation, tester=False):

        for simulation in range(num_simulation):

            if tester:
                self.tester_print(simulation, num_simulation, "MC")
            self.simulator.reset()  # The simulator is already reset for you for each new trajectory

            trajectory = [(self.simulator.state, self.simulator.check_reward())]
            while True:
                next_state = self.make_one_transition(self.default_policy(self.simulator.state))
                if next_state is None:
                    break
                trajectory += [(self.simulator.state, self.simulator.check_reward())]
            for i in range(len(trajectory)):
                R_n = trajectory[i:]
                reward = 0
                for j in range(len(R_n)):
                    reward += R_n[j][1]*DISCOUNT**j
                self.S_MC[trajectory[i][0]] += reward
                self.N_MC[trajectory[i][0]] += 1
                self.MC_values[trajectory[i][0]] = self.S_MC[trajectory[i][0]]/self.N_MC[trajectory[i][0]]

    def TD_run(self, num_simulation, tester=False):

        for simulation in range(num_simulation):

            if tester:
                self.tester_print(simulation, num_simulation, "TD")
            self.simulator.reset()

            s, r, end = self.simulator.state, self.simulator.check_reward(), False
            while True:
                next_s = self.make_one_transition(self.default_policy(s))
                if next_s is None:
                    tdv_next = 0
                    end = True
                else:
                    tdv_next = self.TD_values[next_s]
                self.TD_values[s] = self.TD_values[s] + self.alpha(self.N_TD[s])*(r + DISCOUNT*tdv_next - self.TD_values[s])
                self.N_TD[s] += 1
                s, r = self.simulator.state, self.simulator.check_reward()
                if end:
                    break

    def Q_run(self, num_simulation, tester=False, epsilon=0.4):

        for simulation in range(num_simulation):

            if tester:
                self.tester_print(simulation, num_simulation, "Q")
            self.simulator.reset()

            s, r, end = self.simulator.state, self.simulator.check_reward(), False
            while True:
                a = self.pick_action(s, epsilon)
                next_s = self.make_one_transition(a)
                if next_s is None:
                    q_next = [0,0]
                    end = True
                else:
                    q_next = self.Q_values[next_s]
                self.Q_values[s][a] = self.Q_values[s][a] + self.alpha(self.N_Q[s][a])*(r + DISCOUNT*max(q_next)-self.Q_values[s][a])
                self.N_Q[s][a] += 1
                s, r = self.simulator.state, self.simulator.check_reward()
                if end:
                    break

    def pick_action(self, s, epsilon):
        if random.random() < epsilon:
            return random.choice([HIT, STAND])
        else:
            ht, st = self.Q_values[s][0], self.Q_values[s][1]
            if ht > st:
                return HIT
            return STAND


    def autoplay_decision(self, state):
        hitQ, standQ = self.Q_values[state][HIT], self.Q_values[state][STAND]
        if hitQ > standQ:
            return HIT
        if standQ > hitQ:
            return STAND
        return HIT #Before Q-learning takes effect, just always HIT

    def save(self, filename):
        with open(filename, "w") as file:
            for table in [self.MC_values, self.TD_values, self.Q_values, self.S_MC, self.N_MC, self.N_TD, self.N_Q]:
                for key in table:
                    key_str = str(key).replace(" ", "")
                    entry_str = str(table[key]).replace(" ", "")
                    file.write(f"{key_str} {entry_str}\n")
                file.write("\n")

    def load(self, filename):
        with open(filename) as file:
            text = file.read()
            MC_values_text, TD_values_text, Q_values_text, S_MC_text, N_MC_text, NTD_text, NQ_text, _  = text.split("\n\n")
            
            def extract_key(key_str):
                return tuple([int(x) for x in key_str[1:-1].split(",")])
            
            for table, text in zip(
                [self.MC_values, self.TD_values, self.Q_values, self.S_MC, self.N_MC, self.N_TD, self.N_Q], 
                [MC_values_text, TD_values_text, Q_values_text, S_MC_text, N_MC_text, NTD_text, NQ_text]
            ):
                for line in text.split("\n"):
                    key_str, entry_str = line.split(" ")
                    key = extract_key(key_str)
                    table[key] = eval(entry_str)

    @staticmethod
    def tester_print(i, n, name):
        print(f"\r  {name} {i + 1}/{n}", end="")
        if i == n - 1:
            print()
