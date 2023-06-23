class Evaluation_Metric:
    def __init__(self, metric_name):
        self.metric_name = metric_name

    def compute_metric(self, agents, task_scenario):
        if self.metric_name == "accuracy":
            return self.compute_accuracy(agents, task_scenario)
        elif self.metric_name == "task_completion_time":
            return self.compute_task_completion_time(agents, task_scenario)
        elif self.metric_name == "collaboration_efficiency":
            return self.compute_collaboration_efficiency(agents, task_scenario)
        elif self.metric_name == "conflict_resolution_success":
            return self.compute_conflict_resolution_success(agents, task_scenario)
        else:
            raise ValueError(f"Unsupported metric: {self.metric_name}")

    def compute_accuracy(self, agents, task_scenario):
        correct_answers = 0
        total_answers = 0

        for agent in agents:
            if isinstance(agent, Student):
                correct_answers += agent.correct_answers
                total_answers += agent.total_answers

        if total_answers == 0:
            return 0

        return correct_answers / total_answers

    def compute_task_completion_time(self, agents, task_scenario):
        total_time = 0

        for agent in agents:
            total_time += agent.time_spent_on_task

        return total_time

    def compute_collaboration_efficiency(self, agents, task_scenario):
        total_interactions = 0
        successful_interactions = 0

        for agent in agents:
            total_interactions += agent.total_interactions
            successful_interactions += agent.successful_interactions

        if total_interactions == 0:
            return 0

        return successful_interactions / total_interactions

    def compute_conflict_resolution_success(self, agents, task_scenario):
        total_conflicts = 0
        resolved_conflicts = 0

        for agent in agents:
            if isinstance(agent, Mediator):
                total_conflicts += agent.total_conflicts
                resolved_conflicts += agent.resolved_conflicts

        if total_conflicts == 0:
            return 0

        return resolved_conflicts / total_conflicts