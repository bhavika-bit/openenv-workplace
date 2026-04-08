import os
from openai import OpenAI
from env.environment import WorkplaceEnv, Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
else:
    client = None


def get_action(prompt):
    """
    Returns action using OpenAI if available,
    otherwise uses fallback logic
    """
    try:
        if client:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        else:
            return "default_action"

    except Exception:
        return "default_action"


def run():
    """
    Runs the environment loop and RETURNS result (important for OpenEnv)
    """
    env = WorkplaceEnv()
    obs = env.reset()

    step = 0
    rewards = []
    success = False

    try:
        while True:
            step += 1
            action_str = get_action(obs.content)

            obs, reward, done, info = env.step(Action(response=action_str))

            rewards.append(float(reward))

            if done:
                success = True
                break

    except Exception as e:
        success = False

    finally:
        env.close()

    return {
        "success": success,
        "steps": step,
        "rewards": rewards
    }

if __name__ == "__main__":
    result = run()
    print(result)
