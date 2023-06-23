class Task_Scenario:
    def __init__(self):
        self.environment = None
        self.agent_roles = []
        self.completion_status = False

    def configure_environment(self, environment_config):
        self.environment = environment_config

    def add_agent_role(self, agent_role):
        self.agent_roles.append(agent_role)

    def set_completion_status(self, status):
        self.completion_status = status

    def evaluate_agent_performance(self, agents):
        # Implement evaluation logic based on the specific task scenario
        pass