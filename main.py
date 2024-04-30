import os
import random
from textblob import TextBlob
from yake import KeywordExtractor
from enum import Enum
from collections import defaultdict
import uuid
import requests
import anthropic

import os

# Set the Anthropic API key
os.environ["ANTHROPIC_API_KEY"] = ""

# Rest of the code...

class ThoughtNode:
    def __init__(self, content, parent=None):
        self.content = content
        self.parent = parent
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self

    def is_goal(self):
        # Add logic to check if the thought node represents a goal state
        return False

class SentimentAnalyzer:
    def analyze(self, text):
        blob = TextBlob(text)
        return blob.sentiment.polarity

class Agent:
    def __init__(self, id, goal, name=None, role=None, **kwargs):
        self.id = id
        self.goal = goal
        self.name = name if name else str(uuid.uuid4())
        self.thought_tree = ThoughtNode(content="Root Thought")
        self.role = role
        self.knowledge = set()

        self.client = anthropic.Client(api_key="")
        self.google_api_key = ""

        self.sentiment_analyzer = SentimentAnalyzer()
        self.emotional_state = "neutral"
        self.context_state = defaultdict(lambda: defaultdict(str))
        self.interpersonal_state = defaultdict(str)
        self.knowledge_state = defaultdict(int)
        self.goal_state = "learning"
        self.sentiment_analyzer = SentimentAnalyzer()
        self.keyword_extractor = KeywordExtractor()
        self.thought_tree_explorer = Thought_Tree_Explorer(strategy='depth_first')
        
        for key, value in kwargs.items():
            setattr(self, key, value)

        domain = kwargs.get("domain", None)
        self.experts = kwargs.get('experts', self.generate_experts(domain) if domain else [])

        if role == "expert" and not self.experts:
            raise ValueError("Domain not specified for 'expert' agent.")

    def query_knowledge_graph_api(self, query: str, api_key: str):
        url = f'https://kgsearch.googleapis.com/v1/entities:search?query={query}&key={api_key}&limit=10&indent=True'
        headers = {
            'Accept': 'application/json',
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return data.get('itemListElement', [])
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return []

    def generate_thoughts_with_claude(self, prompt, max_tokens=50):
        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=max_tokens,
            system="You are a helpful assistant.",
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        return response.content[-1].text.strip()

    def generate_task(self, goal, **kwargs):
        task_dict = {
            "goal": goal,
            "steps": self.generate_task_steps(goal),
            "details": None,
        }

        for key, value in kwargs.items():
            task_dict[key] = value

        return task_dict
    
    def generate_task_steps(self, goal):
        prompt = f"Generate a list of 4 steps to accomplish the goal related to {goal}:"
        steps = self.generate_thoughts_with_claude(prompt, max_tokens=100)
        
        if steps:
            return steps.split('; ')
        else:
            return []
    
    def learn_skill(self):
        available_skills = ['A', 'B', 'C', 'D']
        return random.choice(available_skills)

    def is_goal(self):
        return self.goal in self.knowledge

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

    def communicate(self, other_agent):
        prompt = f"Agent {self.id} communicates a task-related message to Agent {other_agent.id} about their shared goal of {self.goal}."
        claude_thoughts = self.generate_thoughts_with_claude(prompt)

        if claude_thoughts:
            self.thought_tree.add_child(ThoughtNode(claude_thoughts))
            return claude_thoughts
        else:
            return None

    def get_context_for_thought_generation(self, other_agent):
        return f"{other_agent.__class__.__name__}: {other_agent.role}"

    def generate_thoughts(self, context):
        thoughts = []
        for expert in self.experts:
            prompt = f"{expert} answers the following question: {context}"
            response = self.query_claude(prompt)
            new_thought = Thought(content=response, parent=self.thought_tree)
            thoughts.append(new_thought)

        return thoughts

    def query_claude(self, prompt):
        response = self.client.completion(
            prompt=f"\n\nHuman: {prompt}\n\nAssistant: ",
            stop_sequences=["\n\nHuman"],
            model="claude-3-opus-20240229",
            max_tokens_to_sample=100,
        )
        return response.completion.strip()

    def evaluate_thoughts(self, thoughts):
        if len(thoughts) == 0:
            return Thought(content="No thoughts generated. Taking a random action.")
        best_thought = random.choice(thoughts)
        return best_thought

    def update_thought_tree(self, best_thought):
        self.thought_tree.children.append(best_thought)
        best_thought.parent = self.thought_tree

    def generate_and_evaluate_thoughts(self, context, method='both'):
        if method == 'tree_exploration':
            best_thought = self.explore_thoughts()
        elif method == 'claude':
            thoughts = self.generate_thoughts(context)
            best_thought = self.evaluate_thoughts(thoughts)
        elif method == 'both':
            thoughts = self.generate_thoughts(context)
            best_thought = self.evaluate_thoughts(thoughts)
        else:
            raise ValueError("Invalid thought generation method")

        return best_thought

    def receive_message(self, other_agent, message):
        self.update_states_based_on_message(message)

    def update_states_based_on_message(self, message):
        ...

    def generate_context_aware_message(self, other_agent, message):
        prompt = f"Generate a relevant and concise task-related response from Agent {self.id} to Agent {other_agent.id} based on the received message '{message}' and their shared goal of {self.goal}."
        claude_response = self.generate_thoughts_with_claude(prompt, max_tokens=100)

        if claude_response:
            return claude_response
        else:
            return None

    def cooperate(self, other_agent):
        ...

    def generate_experts(self, domain, num_experts=5):
        query = f"{domain} experts"
        response = self.query_knowledge_graph_api(query=query, api_key=self.google_api_key)

        if response:
            experts = [item['result']['name'] for item in response[:num_experts]]
            return experts
        else:
            return []
        
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

class Thought_Tree_Explorer:
    def __init__(self, strategy='depth_first'):
        self.strategy = strategy

    def explore_thought_tree(self, agent):
        if self.strategy == 'depth_first':
            return self.depth_first_search(agent.thought_tree_root)
        elif self.strategy == 'breadth_first':
            return self.breadth_first_search(agent.thought_tree_root)
        elif self.strategy == 'heuristic':
            return self.heuristic_search(agent.thought_tree_root)
        else:
            raise ValueError("Invalid exploration strategy")

    def depth_first_search(self, node):
        if node is None:
            return None

        if node.is_goal():
            return node

        for child in node.children:
            result = self.depth_first_search(child)
            if result is not None:
                return result

        return None

    def breadth_first_search(self, node):
        if node is None:
            return None

        queue = [node]

        while queue:
            current_node = queue.pop(0)

            if current_node.is_goal():
                return current_node

            queue.extend(current_node.children)

        return None

    def heuristic_search(self, node):
        if node is None:
            return None

        open_list = [node]
        closed_list = []

        while open_list:
            current_node = min(open_list, key=lambda x: x.heuristic_value())
            open_list.remove(current_node)
            closed_list.append(current_node)

            if current_node.is_goal():
                return current_node

            for child in current_node.children:
                if child not in closed_list:
                    open_list.append(child)

        return None

    def prune_thought_tree(self, node, max_depth):
        if node is None or max_depth < 0:
            return

        if max_depth == 0:
            node.children = []
        else:
            for child in node.children:
                self.prune_thought_tree(child, max_depth - 1)

def display_message(sender, receiver, message, method):
    print(f"Agent {sender.id} ({sender.name}): {message} (method: {method})")
    
    if receiver is not None:
        print(f"Agent {receiver.id} ({receiver.name}) received the message.")
    print()

def main():
    # Initialize TheoryOfMind
    tom = TheoryOfMind()

    # Access environment variables
    user_role = os.getenv('USER_ROLE', 'user1')
    behavior = os.getenv('BEHAVIOR', 'curious')

    # User inputs
    goal = input("Enter the goal for agents: ").strip()
    domain = input("Enter the domain related to the goal: ").strip()

    # Ensuring domain is not empty
    if not domain:
        print("Error: Domain must be specified.")
        return  # Terminate or loop for another input as per your error handling policy

    # More input for unique domains for each expert
    domain1 = input("Enter the first domain related to the goal: ").strip()
    domain2 = input("Enter the second domain related to the goal: ").strip()

    if not domain1 or not domain2:  # Ensure that neither domain string is empty
        print("Error: Both domains must be specified for expert agents.")
        return

    # Create agent instances
    agent1 = Agent(id=1, goal=goal, role='expert', domain=domain1)
    agent2 = Agent(id=2, goal=goal, role='expert', domain=domain2)

    agent1_experts = agent1.generate_experts(domain1)
    agent2_experts = agent2.generate_experts(domain2)

    name1 = agent1_experts[0] if agent1_experts else None
    name2 = agent2_experts[0] if agent2_experts else None

    agent1.name = name1
    agent2.name = name2
    
    task1 = agent1.generate_task(goal, details="detail1")
    task2 = agent2.generate_task(goal, details="detail2")

    # Print tasks derived from the goal for each agent
    print(f"Agent 1 ({agent1.name}) derived task from the goal: {task1}")
    print(f"Agent 2 ({agent2.name}) derived task from the goal: {task2}")
    print()

    # Register agents with TheoryOfMind
    tom.register_agent(agent1)
    tom.register_agent(agent2)

    # Add agents to the Simulation environment
    agents = [agent1, agent2]
    simulation = Simulation(agents)

    # Print introductory message
    print(f"System: A simulated interaction between agents with Theory of Mind is about to start.")
    
    # Execution of the interaction between agents can be implemented with a loop, events, or scenario-based triggers.
    while not simulation.finished:  # Implement a termination condition for the simulation.
        simulation.run_interactions(display_message)
        simulation.update_finished_status()  # Add this line to update the finished status
        # Optionally, update other variables in the simulation loop, e.g., scenario-based triggers.

if __name__ == "__main__":
    main()
