import random
from typing import Dict
import torch as th
from stable_baselines3.common.evaluation import evaluate_policy

from utils.model import train_model


class BaseLines3Neat():
    def __init__(self, model, env, generations=1, population=2, total_timesteps=10_000):
        self.env = env
        self.model = model
        self.gen_size = generations
        self.pop_size = population
        self.total_timesteps = total_timesteps

    def mutate(self, params: Dict[str, th.Tensor], noise_scale: float = 0.01, mutation_probability: float = 0.2) -> Dict[str, th.Tensor]:
        """Mutate parameters by adding scaled normal noise to them with a given probability"""
        mutated_params = {}
        # for name, param in params.items():
        for name, param in params.items():
            if random.random() < mutation_probability:
                mutated_param = param + noise_scale * th.randn_like(param)
                mutated_params[name] = mutated_param
            else:
                mutated_params[name] = param
        return mutated_params

    def create_population(self, pop_size, model_params_list):
        population = model_params_list.copy()

        while len(population) < pop_size:
            population.append(self.mutate(random.choice(model_params_list)))

        return population

    def run(self):
        # Initial Training
        # self.model.learn(total_timesteps=self.total_timesteps)

        model_params = self.model.policy.state_dict()
        population = self.create_population(
            self.pop_size, [model_params])

        top_models_params = [model_params]

        # Keep top 10%
        n_elite = max(2, int(self.pop_size * 0.5))
        # Retrieve the environment
        vec_env = self.model.get_env()
        for iteration in range(self.gen_size):
            population = self.create_population(self.pop_size, top_models_params)
            candidates = []
            for population_i, candidate in enumerate(population):
                print(f"Agent {population_i + 1} < {self.pop_size}")

                # Load new policy parameters to agent.
                # Tell function that it should only update parameters
                # we give it (policy parameters)
                self.model.policy.load_state_dict(candidate, strict=False)
                vec_env.reset()
                self.model.set_env(vec_env)

                # Train the candidate
                train_model(self.model, total_timesteps=self.total_timesteps)

                # Evaluate the candidate
                new_env = self.model.get_env()
                new_env.reset()

                fitness, _ = evaluate_policy(
                    self.model, new_env, n_eval_episodes=5, warn=False
                )
                total_profit = new_env.buf_infos[-1]['total_profit']
                candidates.append((candidate, total_profit, fitness))

            # Take top candiates
            top_candidates = sorted(
                candidates, key=lambda x: x[1] + x[2], reverse=True)[:n_elite]

            # Copy Top Candidate Params to mutate Population
            top_models_params = [sublist[0] for sublist in top_candidates]

            mean_fitness = sum(top_candidate[1]
                               for top_candidate in top_candidates) / n_elite
            print(f"Generation {iteration + 1:<3} Mean top fitness: {mean_fitness:.2f}")
            print(f"Total Profit: {top_candidates[0][1]:.2f}")
            print(f"Total Reward: {top_candidates[0][2]:.2f}")
            print("--------------------------------------")
            self.model.policy.load_state_dict(top_candidates[0][0], strict=False)
            self.model.save(f'neat_best_model_{iteration}')
