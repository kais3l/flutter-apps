# 1.
import torch
import numpy as np
from cowrie.shell.agent import DQNAgent, StateHistoryHelper

# 1. RL Global Initialization (Runs once at Boot to prevent lag spikes)
GLOBAL_HISTORY_N = 5
GLOBAL_RL_AGENT = DQNAgent(state_dim=2 + 6 * GLOBAL_HISTORY_N, action_dim=3)
try:
    GLOBAL_RL_AGENT.load("src/cowrie/shell/dqn_cowrie_model.pth") 
    GLOBAL_RL_AGENT.policy_net.eval()
except Exception as e:
    pass

# 2. String to Neural-Network Categorization Parsing 
def get_command_category(cmd_line):
    if isinstance(cmd_line, bytes):
        cmd = cmd_line.decode('utf-8', errors='ignore').strip().split()[0]
    else:
        cmd = cmd_line.strip().split()[0]
    
    fs_cmds = ['ls', 'cd', 'pwd', 'cat', 'rm', 'mv', 'cp', 'mkdir', 'rmdir', 'touch']
    net_cmds = ['wget', 'curl', 'ssh', 'ftp', 'nc', 'telnet', 'ping']
    sys_cmds = ['uname', 'id', 'whoami', 'ps', 'top', 'free', 'df', 'du']
    exec_cmds = ['chmod', 'chown', 'sudo', 'su', 'sh', 'bash', './']
    edit_cmds = ['vi', 'vim', 'nano', 'echo', 'sed', 'awk']
    
    if cmd in fs_cmds: return 0
    if cmd in net_cmds: return 1
    if cmd in sys_cmds: return 2
    if cmd in exec_cmds: return 3
    if cmd in edit_cmds: return 4
    return 5 


# 2.
    def __init__(self, ...):
        # ... existing cowrie initialization code ...
        
        # RL Agent Connection (Bind to global Singleton)
        self.history_n = GLOBAL_HISTORY_N
        self.history_helper = StateHistoryHelper(n=self.history_n)
        self.rl_agent = GLOBAL_RL_AGENT
        
        # Session Tracking Metrics
        self.session_duration = 0
        self.command_count = 0
        self.last_state = None
        self.last_action = None        


# 3.
    def lineReceived(self, line: str) -> None:
        self.session_duration += 1
        self.command_count += 1
        
        # 1. Update State History using 5-Command sliding window
        cat_idx = get_command_category(line)
        self.history_helper.add_command(cat_idx)
        
        dur = min(self.session_duration / 100.0, 1.0)
        cnt = min(self.command_count / 20.0, 1.0)
        current_state = self.history_helper.get_state(dur, cnt)

        # 2. Get the Agent's Decision
        action = self.rl_agent.get_action(current_state, training=True)
        self.last_state = current_state
        self.last_action = action
        
        # 3. Execute the Action Route
        if action == 2: # BLOCK
            self.protocol.terminal.loseConnection()
            return
        elif action == 1: # DELAY
            from twisted.internet import reactor
            reactor.callLater(2.0, self._internal_lineReceived, line)
            return
        
        # 0: ALLOW Default Bypass
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
            try:
                self.rl_agent.save("src/cowrie/shell/dqn_cowrie_model.pth")
            except Exception:
                pass
