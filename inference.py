import os
from openai import OpenAI
from env.environment import WorkplaceEnv, Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def get_action(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def run():
    env = WorkplaceEnv()
    obs = env.reset()

    print(f"[START] task=workplace env=openenv model={MODEL_NAME}")

    step = 0
    rewards = []
    success = False

    try:
        while True:
            step += 1

            action_str = get_action(obs.content)
            obs, reward, done, info = env.step(Action(response=action_str))

            rewards.append(f"{reward:.2f}")

            print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null")

            if done:
                success = True
                break

    except Exception as e:
        print(f"[STEP] step={step} action=error reward=0.00 done=true error={str(e)}")

    finally:
        env.close()
        print(f"[END] success={str(success).lower()} steps={step} rewards={','.join(rewards)}")


if __name__ == "__main__":
    run()