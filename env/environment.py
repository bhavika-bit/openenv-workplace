from pydantic import BaseModel

class Observation(BaseModel):
    task: str
    content: str

class Action(BaseModel):
    response: str


class WorkplaceEnv:
    def __init__(self):
        self.tasks = [
            ("email", "Mark this email as urgent: 'Server is down'"),
            ("cleaning", "Remove nulls from: [1, None, 2, None, 3]"),
            ("code", "Find bug: for i in range(5): print(i+1)")
        ]
        self.index = 0
        self.done = False

    def reset(self):
        self.index = 0
        self.done = False
        return Observation(task=self.tasks[0][0], content=self.tasks[0][1])

    def step(self, action: Action):
        correct = ["urgent", "[1,2,3]", "no bug"]

        reward = 1.0 if correct[self.index] in action.response.lower() else 0.0

        self.index += 1
        if self.index >= len(self.tasks):
            self.done = True

        obs = None if self.done else Observation(
            task=self.tasks[self.index][0],
            content=self.tasks[self.index][1]
        )

        return obs, reward, self.done, {}

    def state(self):
        return {"index": self.index}

    def close(self):
        pass