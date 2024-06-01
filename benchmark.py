from tqdm.auto import tqdm
import numpy as np
import datasets
from legalbench.evaluation import evaluate
from legalbench.tasks import TASKS, ISSUE_TASKS, RULE_TASKS, CONCLUSION_TASKS, INTERPRETATION_TASKS, RHETORIC_TASKS
from legalbench.evaluation import MANUAL_EVAL_TASKS, evaluate
from legalbench.utils import generate_prompts
from utils import log_text, log_excel_rows
from datetime import datetime
import pandas as pd

class LegalBenchHandler:

    def __init__(self, model_handler):
        self.model_handler = model_handler
        self.dataset_name='nguha/legalbench'
        self.tasks = TASKS
        self.issue_tasks = ISSUE_TASKS
        self.rule_tasks = RULE_TASKS
        self.conclusion_tasks = CONCLUSION_TASKS
        self.interpretation_tasks = INTERPRETATION_TASKS
        self.rhetoric_tasks = RHETORIC_TASKS
        self.base_prompt_path_prefix = './llm-playground/legalbench/tasks/'

    def _load_task_prompts(self, task_name: str, test=True):
        """Load task dataset."""
        partition = 'test'
        if not test: partition = 'train'
        # Load task dataset
        dataset = datasets.load_dataset(self.dataset_name, task_name)
        dataset = dataset[partition].to_pandas()
        # Retrieve prompt template
        with open(f"{self.base_prompt_path_prefix + task_name}/base_prompt.txt") as in_file:
            prompt_template = in_file.read()
        # Build up task prompts
        prompts = generate_prompts(prompt_template=prompt_template, data_df=dataset)

        return dataset, prompts
    
    def _evaluate_task(self, task_name: str, test=True):
        """Run and evaluate task."""
        log_text(self.model_handler.log_model_name, (f'{task_name} task evaluation started with test = {test}.'))
        # Load task dataset and prompts
        dataset, prompts = self._load_task_prompts(task_name, test)
        # Run inference on model to get responses
        responses = [self.model_handler.run_inference(p) for p in prompts]
        if responses == prompts:
            
            log_text(self.model_handler.log_model_name,'SOMETHING IS WRONG: prompts equal to responses.')
            raise ValueError('SOMETHING IS WRONG: prompts equal to responses.')
        log_text(self.model_handler.log_model_name, (f'{len(prompts)} prompts generated for {task_name} task evaluation with test = {test}'))
        # Evaluate responses
        score = evaluate(task_name, responses, dataset['answer'].tolist())
        log_text(self.model_handler.log_model_name, (f'{task_name} evaluation complete with test = {test}'))
        self._log_model_run(task_name, test, score, prompts, dataset['answer'].tolist(), responses)

        return score
    
    def _task_type(self, task_name: str):
        if task_name in self.issue_tasks:
            return 'Issue'
        elif task_name in self.rule_tasks:
            return 'Rule'
        elif task_name in self.conclusion_tasks:
            return 'Conclusion'
        elif task_name in self.interpretation_tasks:
            return 'Interpretation'
        elif task_name in self.rhetoric_tasks:
            return 'Rhetoric'
        else:
            raise ValueError(task_name, ": Unknown category for task name.")
    
    def _log_model_run(self, task_name, test, task_score, prompts, ideal_responses, model_responses):
        """Log model benchmark run results to a CSV file."""
        # Log file
        legalbench_results_log = './llm-playground/log/legalbench_run_log.xlsx'
        # Retrieve task category
        task_category = self._task_type(task_name)
        # Partition from test
        partition = 'test'
        if not test: partition = 'train'
        log_time = str(datetime.now())
        # Prepare data for logging
        rows = []
        for p, ir, mr in zip(prompts, ideal_responses, model_responses):
            row = [
                log_time,
                self.model_handler.log_model_name,
                task_category,
                task_name,
                partition,
                task_score,
                p.replace('\n', ' '),
                ir.replace('\n', ' '),
                mr.replace('\n', ' ')
            ]
            rows.append(row)
        rows_df = pd.DataFrame(rows, columns=[
            'Log Time', 'Model Name', 'Task Category', 'Task Name', 'Partition',
            'Task Score', 'Prompt', 'Ideal Response', 'Model Response'
        ])
        log_excel_rows(self.model_handler.log_model_name, legalbench_results_log, rows_df)
    
    def evaluate_legalbench(self, test=True):
        """Run and evaluate ALL tasks."""
        lb_scores = [self._evaluate_task(t, test) for t in TASKS if (t not in MANUAL_EVAL_TASKS)]
        lb_tot_score = sum(lb_scores) / len(lb_scores)
        print(lb_tot_score)
        return lb_tot_score
    
    



