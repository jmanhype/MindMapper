import os
# import openai  # Commented out to disable the OpenAI-specific dependency
import random
from textblob import TextBlob
from yake import KeywordExtractor
from enum import Enum
from collections import defaultdict
from thought_tree_explorer import Thought_Tree_Explorer  # Add this import at the top
import uuid
import requests

class ThoughtNode:
    def __init__(self, content, parent=None):
        self.content = content
        self.parent = parent
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self

class SentimentAnalyzer:
    def analyze(self, text):
        blob = TextBlob(text)
        return blob.sentiment.polarity
    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self

class Agent:
    def __init__(self, id, goal, name=None, role=None, **kwargs):  # Add **kwargs here
        self.id = id
        self.goal = goal
        self.name = name if name else str(uuid.uuid4())  # Updated name initialization
        self.thought_tree = ThoughtNode(content="Root Thought")
        self.role = role  # Add this line to store the
        self.knowledge = set()  # Add this line to include the 'knowledge' attribute

        self.api_key = "ENTER-OPENAI-KEY"
        # openai.api_key = self.api_key  # Commented since we will not be using OpenAI's API Key
        self.google_api_key = "ENTER-GOOGLE-KEY"

        self.sentiment_analyzer = SentimentAnalyzer()
        self.emotional_state = "neutral"
        self.context_state = defaultdict(lambda: defaultdict(str))
        self.interpersonal_state = defaultdict(str)
        self.knowledge_state = defaultdict(int)
        self.goal_state = "learning"
        self.sentiment_analyzer = SentimentAnalyzer()
        self.keyword_extractor = KeywordExtractor()
        self.thought_tree_explorer = Thought_Tree_Explorer(strategy='depth_first')
        
        # Process any additional keyword arguments
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


    def generate_thoughts_with_gpt3(self, prompt, n=1, max_token=50):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        # response = openai.Completion.create(
        #     engine="text-davinci-003",
        #     prompt=prompt,
        #     n=n,
        #     max_tokens=max_token,
        #     temperature=0.7,
        #     top_p=1,
        #     frequency_penalty=0
        # ) # Comment out all occurrences of this
        return [choice.text.strip() for choice in response.choices]

    # Add this method to the Agent class to generate tasks dynamically based on the goal and kwargs
    def generate_task(self, goal, **kwargs):
        task_dict = {
            "goal": goal,
            "steps": self.generate_task_steps(goal),  # Generate steps for the task
            "details": None,
        }

        for key, value in kwargs.items():
            task_dict[key] = value

        return task_dict
    
    def generate_task_steps(self, goal):
        # Use GPT-3 to generate the steps based on user input
        prompt = f"Generate a list of 4 steps to accomplish the goal related to {goal}:"
        steps = self.generate_thoughts_with_gpt3(prompt, n=1, max_token=100)
        
        if steps:
            return steps[0].split('; ')  # Separate the steps in GPT-3 generated response
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
        # Construct message to communicate using the Anthropi API
        prompt = f"Agent {self.id} communicates a task-related message to Agent {other_agent.id} about their shared goal of {self.goal}."
        anthropic_thoughts = self.send_message_with_anthropic(prompt)
        
        # Replace OpenAI functionality with Anthropic usage
        if anthropic_thoughts:
            self.thought_tree.add_child(ThoughtNode(anthropic_thoughts[0]))
            return anthropic_thoughts[0]
        else:
            return None

    def get_context_for_thought_generation(self, other_agent):
        return f"{other_agent.__class__.__name__}: {other_agent.role}"

    def generate_thoughts(self, context):
        thoughts = []
        for expert in self.experts:
            prompt = f"{expert} answers the following question: {context}"
            response = self.query_openai_api(prompt)
            new_thought = Thought(content=response, parent=self.thought_tree)
            thoughts.append(new_thought)

        return thoughts

    def query_openai_api(self, prompt):
        response = openai.Completion.create(
            engine="davinci-codex",
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].text.strip()

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
        elif method == 'gpt3':
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
        # Use the Anthropi API to respond to the message contextually
        prompt = 'Generate relevant and concise response...'
        anthropic_response = self.send_message_with_anthropic(prompt)
        
        if anthropic_response:
            return anthropic_response[0]
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

    def send_message_with_anthropic(self, prompt):
        import os
        import requests

        # Set up authentication and headers
        headers = {
            "Authorization": f"Bearer {os.getenv('ANTHROPIC_API_KEY')}",
            "Content-Type": "application/json"
        }

        # Create the request payload following Anthropic's API requirements
        data = {
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        # Set up authentication and headers
        headers = {
            "Authorization": f"Bearer {os.getenv('ANTHROPIC_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        # Create the request payload following Anthropic's API requirements
        data = {
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        try:
            # Send the POST request to the configured endpoint
            response = requests.post("https://api.anthropic.com/completion", json=data, headers=headers)
            response.raise_for_status()  # Throws an exception for HTTP errors like 4xx or 5xx
            if response.status_code == 200:
                api_response = response.json()
                messages = [choice["text"].strip() for choice in api_response.get('choices', []) if 'text' in choice]
                return messages
            elif response.status_code == 429:
                print("Error: Rate limit exceeded")
            elif response.status_code == 401:
                print("Error: Unauthorized - check your API key")
            elif response.status_code == 403:
                print("Error: Forbidden - access denied")
            elif response.status_code == 404:
                print("Error: Not Found - the requested resource does not exist")
            elif response.status_code == 500:
                print("Error: Internal Server Error - something has gone wrong on the web server")
            else:
                print(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"HTTP Request failed: {e}")
            return []
        
if __name__ == "__main__":
    agent1 = Agent(1, "Goal A")
    agent2 = Agent(2, "Goal B")

    agent1.communicate(agent2)
