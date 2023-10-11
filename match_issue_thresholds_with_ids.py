import pandas as pd
from sqlalchemy import create_engine, text as sqlalchemy_text
from psql_info import (
        PSQL_USER,
        DATABASE_PROD_READ_REPLICA,
        HOST_PROD_READ_REPLICA,
        PASSWORD_PROD_READ_REPLICA
)

def get_psql_engine(user, password, host, database):
    print(user, password, host, database)
    return create_engine(f"postgresql://{user}:{password}@{host}:5432/{database}")

def get_issue_types(engine, key_name):
    connection = engine.connect()
    key_name = key_name.upper()
    query = f"""
        SELECT
          sit.issue_type_key
          ,nsig.node_type
          ,nsit.id AS issue_type_id
        FROM
          public.service_issue_types AS sit
          INNER JOIN public.node_service_issue_types AS nsit
            ON nsit.service_issue_type_id = sit.id
          INNER JOIN public.node_service_issue_groups AS nsig
            ON nsig.id = nsit.node_service_issue_group_id
        WHERE nsig.node_type LIKE '%TRP%'
         AND sit.issue_type_key LIKE '%{ key_name }%';
    """
    return pd.read_sql(sqlalchemy_text(query), con=engine)

def get_query(label, value, node_type, issue_type_id):
    query = f"""
        SELECT
          { issue_type_id } AS issue_type_id,
          '{ node_type }' AS node_type,
          UPPER('{ label }') AS threshold_name,
          '{ value }' AS threshold_value,
          'INTEGER' AS threshold_value_data
    """
    return query

buffer_hours = 12
lst = [
#       ('contains_animal',1),
#       ('contains_branch',1),
#       ('damaged',1),
#       ('liner_issue',1),
#       ('liner_full',1),
#       ('missing_lure',1),
#       ('blurry_physical_issue',3),
#       ('camera_zoomed_out',1),
#       ('camera_misaligned_high_priority',1)
       ('contains_animal',buffer_hours),
       ('contains_branch',buffer_hours),
       ('damaged',buffer_hours),
       ('liner_issue',buffer_hours),
       ('liner_full',buffer_hours),
       ('missing_lure',buffer_hours),
       ('blurry_physical_issue',buffer_hours),
       ('camera_zoomed_out',buffer_hours),
       ('camera_misaligned_high_priority',buffer_hours)
]
DATABASE = DATABASE_PROD_READ_REPLICA
HOST = HOST_PROD_READ_REPLICA
PASSWORD = PASSWORD_PROD_READ_REPLICA
USER = PSQL_USER
engine = get_psql_engine(USER, PASSWORD, HOST, DATABASE)
final_query = ""
for i, info in enumerate(lst):
    label, value = info
    df = get_issue_types(engine,label)
    for j, row in df.iterrows():
        final_query += get_query(
                #f'consecutive_{label}_images', 
                f'hours_after_install', 
                value, 
                row.node_type, 
                row.issue_type_id
        )
        if (i < len(lst) - 1) or (j<len(df) - 1):
            final_query += "    UNION ALL "
print(final_query)
with open("update_new_thresholds_query.sql", "w") as text_file:
    text_file.write(final_query)
