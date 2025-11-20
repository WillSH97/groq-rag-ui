'''
    RAGChat - a GUI for the quick development of RAG chatbots, used to
    teach the basic intuitions behind RAG LLMs.
    
    Copyright (C) 2025 QUT GenAI Lab

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    For more information, contact QUT GenAI lab via: genailab@qut.edu.au
'''


from chromadb_engine import (
    client,
    switch_db,
    make_db_from_pdf,
    make_db_from_csv,
    make_db_from_docx,
    make_db_from_txt,
)

from copy import deepcopy

def create_injection_prompt(
    db_name,
    input_msg,
    num_return,
    max_dist: float = None,
    inject_col: str = None,
    inject_template: str = "{INJECT_TEXT}, {USER_MESSAGE}",
):
    """
    creates injection prompt based on input msg and query return.
    """
    collection = client.get_collection(db_name)
    if isinstance(input_msg, list): #for bullshit llama 4 img format
        query = input_msg[0]["text"]
    else:
        query = input_msg
    results = collection.query(query_texts=[query], n_results=num_return)

    if inject_col:
        inject_list = [x[inject_col] for x in results["metadatas"][0]]

    else:
        inject_list = results["documents"][0]

    injection_str = "\n\n\n".join(inject_list)
    augmented_user_msg = inject_template.format(
        INJECT_TEXT=injection_str, USER_MESSAGE=query
    )
    if isinstance(input_msg, list):
        output_msg = deepcopy(input_msg)
        output_msg[0]["text"] = augmented_user_msg
        return output_msg
    else:
        return augmented_user_msg
