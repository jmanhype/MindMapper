class Simulation:
    def __init__(self, agents):
        self.agents = agents
        self.data_log = []
        self.finished = False  # Add this attribute to the Simulation class

    # Add this method to update the finished status when the termination condition is met
    def update_finished_status(self):
        # Implement termination condition checking logic here
        # For example, you can check if all agents have reached their goals
        all_agents_reached_goal = all([agent.is_goal() for agent in self.agents])

        if all_agents_reached_goal:
            self.finished = True

    def run_interactions(self, display_message):
        for agent in self.agents:
            for other_agent in self.agents:
                if agent != other_agent:
                    method = 'both'  # Change this to use different thought generation methods
                    message = agent.communicate(other_agent)
                    other_agent.receive_message(agent, message)
                    self.data_log.append((agent, other_agent, message))

                    if display_message:
                        display_message(agent, other_agent, message, method)

                    response = other_agent.generate_context_aware_message(agent, message)
                    agent.receive_message(other_agent, response)
                    self.data_log.append((other_agent, agent, response))

                    if display_message:
                        display_message(other_agent, agent, response, method)

                    agent.cooperate(other_agent)
                    other_agent.cooperate(agent)

                    agent.cooperate(other_agent)
                    other_agent.cooperate(agent)