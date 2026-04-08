# 1.
import torch
import numpy as np
from cowrie.shell.agent import DQNAgent, StateHistoryHelper

# 2.
    def __init__(self, ...):
        # ... existing cowrie initialization code ...
        
        # RL Agent Initialization
        self.history_n = 5
        self.history_helper = StateHistoryHelper(n=self.history_n)
        
        # Dynamic State Dimension (2 Metrics + N Historical Categorizations)
        self.rl_agent = DQNAgent(state_dim=2 + 6 * self.history_n, action_dim=3)
        self.rl_agent.load("src/cowrie/shell/dqn_cowrie_model.pth") 
        self.rl_agent.policy_net.eval()
        
        # Session Metrics
        self.session_duration = 0
        self.command_count = 0
        self.last_state = None
        self.last_action = None        

# 3.

    def lineReceived(self, line: str) -> None:
        self.session_duration += 1
        self.command_count += 1
        
        # 1. Evaluate the State
        cat_idx = get_command_category(line)
        self.history_helper.add_command(cat_idx)
        
        dur = min(self.session_duration / 100.0, 1.0)
        cnt = min(self.command_count / 20.0, 1.0)
        current_state = self.history_helper.get_state(dur, cnt)

        # 2. Get the Agent's Decision
        # Epsilon can be tweaked here for exploitation vs exploration
        action = self.rl_agent.get_action(current_state, training=True)
        self.last_state = current_state
        self.last_action = action
        
        # 3. Execute the Action 
        if action == 2: # BLOCK
            self.protocol.terminal.loseConnection()
            return
        elif action == 1: # DELAY
            from twisted.internet import reactor
            # Delays execution by 2.0 seconds
            reactor.callLater(2.0, self._internal_lineReceived, line)
            return
        
        # 0: ALLOW - Standard processing
        self._internal_lineReceived(line)

# 4.
    def connectionLost(self, reason):
        if self.last_state is not None:
            # Reward: +10 if session has good duration, -10 if attacker fled early
            final_reward = 10.0 if self.command_count > 10 else -10.0
            
            # Record terminal transition and update weights
            self.rl_agent.memory.push(
                self.last_state, 
                self.last_action, 
                final_reward, 
                self.last_state, 
                True
            )
            self.rl_agent.update()
            
            # Save the "Smarter" weights to disk
            self.rl_agent.save("src/cowrie/shell/dqn_cowrie_model.pth")
