import random
from enum import Enum
from thought_tree_explorer import Thought_Tree_Explorer

class TheoryOfMind:
    def __init__(self):
        self.agents = []

    def register_agent(self, agent):
        self.agents.append(agent)

    def update_agents(self):
        for agent in self.agents:
            agent.update()

    def agent_interaction(self, agent_a, agent_b):
        shared_experience = agent_a.observe(agent_b)
        understanding = agent_a.understand(shared_experience)
        agent_b.receive_understanding(understanding)

class ExperienceType(Enum):
    POSITIVE = 1
    NEUTRAL = 2
    NEGATIVE = 3

class Experience:
    def __init__(self, experience_type, content, **kwargs):
        self.experience_type = experience_type
        self.content = content
        self.attributes = kwargs

class Thought:
    def __init__(self, content, parent=None):
        self.content = content
        self.parent = parent
        self.children = []

class Agent:
    def __init__(self, id, theory_of_mind, goal, strategy='depth_first'):
        self.id = id
        self.theory_of_mind = theory_of_mind
        self.goal = goal
        self.knowledge = set()
        self.experiences = []
        self.thought_tree_root = Thought(content="Root Thought")
        self.thought_tree_explorer = Thought_Tree_Explorer(strategy)

    def update(self):
        best_thought = self.explore_thoughts()
        learned_skill = best_thought.content
        self.knowledge.add(learned_skill)
        experience_type = ExperienceType.POSITIVE if learned_skill == self.goal else ExperienceType.NEUTRAL
        self.experiences.append(Experience(experience_type, f"Learned {learned_skill}"))

    def explore_thoughts(self):
        return self.thought_tree_explorer.explore_thought_tree(self)

    def is_goal(self):
        return self.thought_tree_root.is_goal()

    def learn_skill(self):
        available_skills = ['A', 'B', 'C', 'D']
        return random.choice(available_skills)

    def observe(self, other_agent):
        return random.choice(other_agent.get_experiences())

    def understand(self, shared_experience):
        if shared_experience.experience_type == ExperienceType.POSITIVE:
            self.knowledge.add(shared_experience.content.split(' ')[-1])

    def receive_understanding(self, understanding):
        pass

    def get_experiences(self):
        return self.experiences

    def __str__(self):
        return f"[Agent {self.id}] - knowledge: {', '.join(list(self.knowledge))}; goal: {self.goal}"


if __name__ == "__main__":
    tom = TheoryOfMind()
    goal = 'C'

    agent1 = Agent(1, tom, goal)
    agent2 = Agent(2, tom, goal)

    tom.register_agent(agent1)
    tom.register_agent(agent2)

    max_iterations = 50
    current_iteration = 1

    while not (agent1.is_goal() and agent2.is_goal()) and current_iteration <= max_iterations:
        tom.update_agents()
        tom.agent_interaction(agent1, agent2)
        print(f"Iteration {current_iteration}:")
        print(agent1)
        print(agent2)
        print()
        current_iteration += 1