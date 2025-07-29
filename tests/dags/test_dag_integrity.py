import pytest
from airflow.models import DagBag
import os


class TestDagIntegrity:
    
    @pytest.fixture
    def dagbag(self):
        """Create a DagBag for testing"""
        dag_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'dags')
        return DagBag(dag_folder=dag_folder, include_examples=False)
    
    def test_dag_bag_import(self, dagbag):
        """Test that DagBag imports successfully with no errors"""
        assert dagbag.import_errors == {}, f"DAG import errors: {dagbag.import_errors}"
    
    def test_dag_loaded(self, dagbag):
        """Test that all expected DAGs are loaded"""
        expected_dags = [
            'sales_forecast_training',
            'sales_forecast_retraining',
            'sales_forecast_inference'
        ]
        
        for dag_id in expected_dags:
            assert dag_id in dagbag.dags, f"DAG {dag_id} not found in DagBag"
    
    def test_dag_structure(self, dagbag):
        """Test DAG structure and dependencies"""
        # Test training DAG
        training_dag = dagbag.get_dag('sales_forecast_training')
        assert training_dag is not None
        
        # Check task count
        assert len(training_dag.tasks) >= 7, "Training DAG should have at least 7 tasks"
        
        # Check critical tasks exist
        task_ids = [task.task_id for task in training_dag.tasks]
        expected_tasks = ['extract_data', 'validate_data', 'train_models', 'evaluate_models']
        for task_id in expected_tasks:
            assert task_id in task_ids, f"Task {task_id} not found in training DAG"
        
        # Check dependencies
        extract_task = training_dag.get_task('extract_data')
        validate_task = training_dag.get_task('validate_data')
        assert validate_task in extract_task.downstream_list, "validate_data should be downstream of extract_data"
    
    def test_dag_schedule_interval(self, dagbag):
        """Test that DAGs have appropriate schedule intervals"""
        dag_schedules = {
            'sales_forecast_training': '@weekly',
            'sales_forecast_retraining': '@daily',
            'sales_forecast_inference': '@daily'
        }
        
        for dag_id, expected_schedule in dag_schedules.items():
            dag = dagbag.get_dag(dag_id)
            assert dag.schedule_interval == expected_schedule, \
                f"DAG {dag_id} has schedule {dag.schedule_interval}, expected {expected_schedule}"
    
    def test_dag_tags(self, dagbag):
        """Test that DAGs have appropriate tags"""
        for dag_id in ['sales_forecast_training', 'sales_forecast_retraining', 'sales_forecast_inference']:
            dag = dagbag.get_dag(dag_id)
            assert 'ml' in dag.tags, f"DAG {dag_id} should have 'ml' tag"
    
    def test_no_import_errors(self, dagbag):
        """Test that there are no import errors in any DAG"""
        assert len(dagbag.import_errors) == 0, \
            f"Found import errors: {dagbag.import_errors}"
    
    def test_dag_retries(self, dagbag):
        """Test that all DAGs have retry configuration"""
        for dag_id, dag in dagbag.dags.items():
            assert dag.default_args.get('retries', 0) >= 1, \
                f"DAG {dag_id} should have at least 1 retry configured"
    
    def test_dag_emails(self, dagbag):
        """Test that all DAGs have email configuration"""
        for dag_id, dag in dagbag.dags.items():
            assert 'email' in dag.default_args, \
                f"DAG {dag_id} should have email configuration"
            assert dag.default_args.get('email_on_failure', False) == True, \
                f"DAG {dag_id} should have email_on_failure enabled"