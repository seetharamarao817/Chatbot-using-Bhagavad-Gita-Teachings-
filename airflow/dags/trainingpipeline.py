from __future__ import annotations
import json
#from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from pipelines.training_pipeline import TrainingPipeline

training_pipeline=TrainingPipeline()

with DAG(
    "chatbot_training_pipeline",
    default_args={"retries": 2},
    description="Continous training for the chatbot",
    schedule="@weekly",
    start_date=pendulum.datetime(2024,2,15, tz="UTC"),
    catchup=False,
    tags=["Deep_learning ","Encoder and decoder","chatbot"],
) as dag:
    
    dag.doc_md = __doc__
    
    def data_ingestion(**kwargs):
        ti = kwargs["ti"]
        questions,answers=training_pipeline.start_data_ingestion()
        ti.xcom_push("Input_data", {"Questions":questions,"Answers":answers})


    def data_preprocessing(**kwargs):
        ti = kwargs["ti"]
        input_data = ti.xcom_pull(task_ids="data_ingestion",key='Input_data')
        tokenizer,vocab_size=training_pipeline.start_data_preprocessing(input_data["Questions"],input_data["Answers"])
        ti.xcom_push("preprocessing_artifact", {"Tokenizer":tokenizer,"Vocab_size":vocab_size})

    def data_transformations(**kwargs):
        ti = kwargs["ti"]
        input_data = ti.xcom_pull(task_ids="data_ingestion",key='Input_data')
        preprocessing_data = ti.xcom_pull(task_ids="data_preprocessing",key='preprocessing_artifact')
        encoder_input, decoder_input, decoder_output, max_len_q, max_len_a =training_pipeline.start_data_transformation(preprocessing_data["Tokenizer"],
            input_data["Questions"],input_data["Answers"],preprocessing_data["Vocab_size"])
        
        ti.xcom_push("data_transformations_artifcat", {"encoder_input":encoder_input, 
                              "decoder_input":decoder_input,"decoder_output":decoder_output,"max_len_q":max_len_q,"max_len_a":max_len_a})


    def load_and_train_model(**kwargs):
        ti = kwargs["ti"]
        transformations_data = ti.xcom_pull(task_ids="data_transformations", key='data_transformations_artifact')

        # Load the existing model or create a new one
        model = training_pipeline.model_building(
            vocab_size=transformations_data["max_len_q"],
            embedding_dim=256,  # Adjust these parameters as needed
            hidden_units=512,
            maxlen_questions=transformations_data["max_len_q"],
            maxlen_answers=transformations_data["max_len_a"]
        )

        # Start training the model
        training_pipeline.model_training(
            model,
            transformations_data["encoder_input"],
            transformations_data["decoder_input"],
            transformations_data["decoder_output"],
            batch_size=64, 
            epochs=50,
            validation_split=0.18
        )

        
     # Define tasks
    task_ingestion = PythonOperator(
        task_id='data_ingestion',
        python_callable=data_ingestion,
        dag=dag,
    )

    task_preprocessing = PythonOperator(
        task_id='data_preprocessing',
        python_callable=data_preprocessing,
        provide_context=True,
        dag=dag,
    )

    task_transformations = PythonOperator(
        task_id='data_transformations',
        python_callable=data_transformations,
        provide_context=True,
        dag=dag,
    )

    task_load_and_train_model = PythonOperator(
        task_id='load_and_train_model',
        python_callable=load_and_train_model,
        provide_context=True,
        dag=dag,
    )

    
    
# Set the task dependencies
task_ingestion >> task_preprocessing >> task_transformations >> task_load_and_train_model    
    
   


