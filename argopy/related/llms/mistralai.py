import importlib
import os
import getpass
import pandas as pd
from typing import Literal
import logging

from argopy import __version__
from argopy.errors import DataNotFound, InvalidDataset
from argopy.stores import httpstore
from argopy.options import OPTIONS
from argopy import ArgoDocs


log = logging.getLogger("argopy.related.mistralai")


has_ipython = (spec := importlib.util.find_spec("IPython")) is not None
if has_ipython:
    from IPython.display import display, Markdown

try:
    importlib.import_module("mistralai")  # noqa: E402
    from mistralai import Mistral
    from mistralai import models as MistralModels

except ImportError:
    log.debug("MistralAI not installed, chatbot not available")
    pass

PREF = "\033["
RESET = f"{PREF}0m"

class COLORS:
    black = "30m"
    red = "31m"
    green = "32m"
    yellow = "33m"
    blue = "34m"
    magenta = "35m"
    cyan = "36m"
    white = "37m"


class Assistant:
    """A chatbot based on MistralAI LLM and informed with the Argo documentation

    With this class, we experiment with the MistralAI LLMs, using the `mistralai library <https://docs.mistral.ai/getting-started/clients/>`_

    The strategy is to upload argopy and Argo manuals to an online library, and to inform an agent with these documents (with OCR).

    A conversation with the agent can then be initiated to get reliable information about Argo.

    Examples
    --------
    .. code-block:: python

        from argopy.related.mistralai import Assistant

        bot = Assistant()

        bot.agent_update()

        bot.library_upload()
        bot.library_describe()
        bot.library_to_dataframe()

        bot.chat()
        bot.replay()

    .. code-block:: bash
        :caption: Chatbot from the command line

        python -c "from argopy.related.mistralai import Assistant; Assistant().agent_update().chat()"

        python -c "from argopy.related.mistralai import Assistant; Assistant().replay()"

        python -c "from argopy.related.mistralai import Assistant; Assistant(name='test', informed_agent=False).agent_update().chat()"

        python -c "from argopy.related.mistralai import Assistant; Assistant(name='test', informed_agent=False).agent_update().replay()"

    ⚠️ Limitations
    Please note that, like any AI, the model may occasionally generate an inaccurate or imprecise answer. Always refer
    to the provided sources to verify the validity of the information given.

    ⚠️ Limitations
    Some limitations arise for free account, most notably the limitation to 20 documents.

    Notes
    -----
    https://docs.mistral.ai/getting-started/quickstart/#account-setup
    https://console.mistral.ai/api-keys
    https://github.com/mistralai/client-python
    https://docs.mistral.ai/api/
    """

    def __init__(self, name : str = "argo", model : Literal["mistral-medium-2505", "devstral-small-2507"] = "mistral-medium-2505", informed_agent : bool = True, **kwargs):
        if OPTIONS["mistral_api_key"] is None:
            if os.getenv("MISTRAL_API_KEY") is None:
                raise ValueError(
                    "You must specify a valid Mistral API key (with an environment variable 'MISTRAL_API_KEY'. If you don't have an API key, follow instructions at: https://docs.mistral.ai/getting-started/quickstart/#account-setup"
                )
            else:
                api_key = os.getenv("MISTRAL_API_KEY")
        else:
            api_key = OPTIONS["mistral_api_key"]
        self._api_key = api_key

        self.library_name = f"{name}-library"
        self.agent_name = f"{name}-agent"
        self.username = getpass.getuser()
        self.model = model
        self.informed_agent = informed_agent

        self.hline = "<hr>" if self.runner == "notebook" else "-"*os.get_terminal_size().columns

    def __repr__(self):
        txt = [f"<argopy.MistralAIAgent.{self.agent_name}>"]
        txt.append(f"🔗 {self.agent_uri}")
        try:
            docs = self.library_to_dataframe()
            txt.append(f"📚 '{self.library_name}' has {self.library_to_dataframe().shape[0]} document(s):")
            txt.append("\n".join([f"\t- {row}" for row in docs['name']]))
        except InvalidDataset:
            txt.append(f"📚 '{self.library_name}' has not been created yet ! You can use the 'library_upload()' method to create and upload content.")
        except DataNotFound:
            txt.append(f"No document found in the library '{self.library_name}', please use the 'library_upload()' method to upload content.")

        return "\n".join(txt)

    @property
    def runner(self) -> str:
        try:
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                return "notebook"  # Jupyter notebook or qtconsole
            elif shell == "TerminalInteractiveShell":
                return "terminal"  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return "standard"  # Probably standard Python interpreter

    def print_line(self, content : str = "", role : str = "") -> None:
        if self.runner == "notebook":
            if role != "":
                display(Markdown("**%s** %s" % (role, content)))
            else:
                display(Markdown(content))
        else:
            if role != "":
                txt = (
                    f"{PREF}{1};{COLORS.yellow}"
                    + role
                    + " "
                    + RESET
                    + f"{PREF}{0};{COLORS.cyan}"
                    + content
                    + " "
                    + RESET
                )
            else:
                txt = (
                        f"{PREF}{0};{COLORS.cyan}"
                        + content
                        + " "
                        + RESET
                )
            print(txt)

    def print_source(self, content : str, summary : str ="Details"):
        if self.runner == "notebook":
            display(Markdown(f"<details><summary>{summary}</summary><br><small>{content}</small></details>"))
        else:
            txt = (
                    f"{PREF}{1};{COLORS.yellow}"
                    + summary
                    + " "
                    + RESET
                    + f"{PREF}{0};{COLORS.red}"
                    + content
                    + " "
                    + RESET
            )
            print(f"\n{txt}")

    def print_message_outputs(self, outputs):
        for message in outputs:
            if isinstance(message, MistralModels.MessageOutputEntry):
                if isinstance(message.content, str):
                    self.print_line(f"\n{message.content}", "🤖")

                if isinstance(message.content, list):
                    for chunk in message.content:
                        if chunk.type == 'text':
                            self.print_line(chunk.text, "🤖")
                        if chunk.type == 'tool_reference':
                            if chunk.tool == 'web_search':
                                if self.runner == "notebook":
                                    title = "Source"
                                    content = f"<a href='{chunk.url}'>{chunk.title}</a><br><small>{chunk.description}</small>"
                                else:
                                    title = "Source:"
                                    content = f"{chunk.title}: {chunk.url}\n{chunk.description}"
                                self.print_source(content, title)

    @property
    def library_id(self):
        if getattr(self, "_lib_id", None) is None:
            self.library_create()
        return self._lib_id

    def library_create(self):
        """Get the library ID, create if necessary

        https://github.com/mistralai/client-python/blob/main/docs/sdks/libraries/README.md

        Returns
        -------
        str
        """
        with Mistral(
            api_key=self._api_key,
        ) as mistral:

            res = mistral.beta.libraries.list()

            exist = False
            for lib in res.data:
                if lib.name == self.library_name:
                    exist = True
                    mylib_id = lib.id

            # if exist:
            #     res = mistral.beta.libraries.get(library_id=mylib_id)
            # print(res.generated_description)
            # print(f"{res.nb_documents} document(s) in this library:")

            if not exist:
                res = mistral.beta.libraries.create(
                    name=self.library_name,
                    description="A set of documents about Argo and the argopy library",
                )
                # print(res)
                mylib_id = res.id

        self._lib_id = mylib_id
        return self._lib_id

    def library_upload(self, alldocs: bool =False):
        """
        https://github.com/mistralai/client-python/blob/main/docs/sdks/documents/README.md

        Parameters
        ----------
        alldocs: bool, default = False
            If set to False, the argopy cheatsheet and the Argo user's manual are uploaded.

            If set to True, the argopy cheatsheet and all the Argo documentation are uploaded (28 PDFs, 1042 pages as of Oct.8th 2025).

        Returns
        -------
        None

        Notes
        -----
        The full list of Argo documents is obtained from :class:`ArgoDocs`.
        """

        list_documents = {
            f"Argopy {__version__} cheatsheet": "https://argopy.readthedocs.io/en/latest/_static/argopy-cheatsheet.pdf",
        }
        if alldocs:
            df = ArgoDocs().list
            for ii, row in df.iterrows():
                list_documents.update({row['title']: ArgoDocs(row['id']).pdf.split(';')[-1].strip()})
        else:
            list_documents.update({"Argo user's manual": ArgoDocs(29825).pdf.split(';')[-1]})

        with Mistral(
            api_key=self._api_key,
        ) as mistral:

            res = mistral.beta.libraries.documents.list(
                library_id=self.library_id,
                page_size=100,
                page=0,
                sort_by="created_at",
                sort_order="desc",
            )

            for doc_name, doc_uri in list_documents.items():
                exist = False
                if getattr(res, 'data', False):
                    for lib_doc in res.data:
                        if lib_doc.name == doc_name:
                            exist = True

                if not exist:
                    with httpstore().open(doc_uri) as doc:
                        res = mistral.beta.libraries.documents.upload(
                            library_id=self.library_id,
                            file={"file_name": doc_name, "content": doc.read()},
                        )
                        print(f"{doc_name} uploaded to library {self.library_id} with success. Wait for some time for it to be processed.")
        return self

    def library_describe(self):

        with Mistral(
            api_key=self._api_key,
        ) as mistral:

            res = mistral.beta.libraries.list()

            exist = False
            for lib in res.data:
                if lib.name == self.library_name:
                    exist = True

            if exist:
                res = mistral.beta.libraries.get(library_id=self.library_id)
                print(f"Generated Description: {res.generated_description}")
                print(f"{res.nb_documents} document(s) in this library:")

                res = mistral.beta.libraries.documents.list(
                    library_id=self.library_id,
                    page_size=100,
                    page=0,
                    sort_by="created_at",
                    sort_order="desc",
                )
                for doc in res.data:
                    print(
                        f"\n- {doc.name} with {doc.number_of_pages} pages ({doc.size} bytes - {doc.processing_status})"
                    )
                    print(f"  {doc.summary}")

    def library_to_dataframe(self, update=False, errors='raise'):
        if update or getattr(self, '_library_df', None) is None:

            with Mistral(
                api_key=self._api_key,
            ) as mistral:

                res = mistral.beta.libraries.list()

                exist = False
                for lib in res.data:
                    if lib.name == self.library_name:
                        exist = True

                if exist:
                    res = mistral.beta.libraries.get(library_id=self.library_id)
                    # print(f"Generated Description: {res.generated_description}")
                    # print(f"{res.nb_documents} document(s) in this library:")

                    res = mistral.beta.libraries.documents.list(
                        library_id=self.library_id,
                        page_size=100,
                        page=0,
                        sort_by="created_at",
                        sort_order="desc",
                    )
                    documents = []
                    for doc in res.data:
                        documents.append([doc.name, doc.number_of_pages, doc.size, doc.processing_status, doc.summary])
                    df = pd.DataFrame(documents, columns=["name", "number_of_pages", "size", "processing_status", "summary"])
                    if df.shape[0] == 0:
                        if errors == 'raise':
                            raise DataNotFound("No document in this library")
                        else:
                            return None
                    else:
                        self._library_df = df
                elif errors == 'raise':
                    raise InvalidDataset("This library does not exist")
                else:
                    return None

        return self._library_df

    @property
    def agent_id(self):
        if getattr(self, "_agent_id", None) is None:
            self.agent_create()
        return self._agent_id

    def agent_create(self):
        """Get the agent ID, create if necessary

        https://github.com/mistralai/client-python/blob/main/docs/sdks/mistralagents/README.md

        Returns
        -------
        str
        """

        with Mistral(
            api_key=self._api_key,
        ) as mistral:

            res = mistral.beta.agents.list()

            exist = False
            for agent in res:
                if agent.name == self.agent_name:
                    exist = True
                    self._agent_id = agent.id

            if not exist:
                res = mistral.beta.agents.create(model=self.model, name=self.agent_name)
                log.info(f"MistralAI agent {self.agent_name} created")
                self._agent_id = res.id
                self.agent_update()

        return self._agent_id

    def agent_update(self):

        tools = [{"type": "web_search"}]
        prompt = "Use the web search tool to access external information. you can get preferred information from the following websites: https://argopy.readthedocs.io; https://github.com/euroargodev; https://github.com/OneArgo; https://github.com/ArgoRTQC; https://github.com/ArgoDMQC; https://argo.ucsd.edu; http://www.argodatamgt.org; https://www.ocean-ops.org; http://www.argodatamgt.org/Documentation; https://vocab.nerc.ac.uk."

        if self.informed_agent:
            tools.append({"type": "document_library", "library_ids": [self.library_id]})
            prompt += " Also use the library tool to access external documents."

        prompt += " If code is requested, just answer code without install instructions and always prefer code examples based on the Argopy library wherever relevant. Assume the role of a research scientist in oceanography working on Argo floats data and programming in Python. For all your answers you will provide a 'Answer reliability scale' from 1 (hypothetic) to 5 (highly reliable) to indicate how confident your are in your answer."

        with Mistral(
            api_key=self._api_key,
        ) as mistral:

            res = mistral.beta.agents.update(
                agent_id=self.agent_id,
                description="An expert agent fully aware of Argo documents",
                instructions=prompt,
                tools=tools,
                completion_args={
                    "temperature": 0.3,
                    "top_p": 0.95,
                },
            )
            log.info(f"MistralAI agent {self.agent_name} updated")
        return self

    @property
    def agent_uri(self):
        return f"https://console.mistral.ai/build/agents/{self.agent_id}"

    def _conversation_tell(self, txt):
        new = getattr(self, "_current_conversation_id", False) is False
        # self.print_line(f" This a new conversation: {new}", "🤖")

        with Mistral(
            api_key=self._api_key,
        ) as mistral:

            if new:
                res = mistral.beta.conversations.start(
                    agent_id=self.agent_id,
                    name="A new conversation about Argo",
                    inputs=txt,
                )
                self._current_conversation_id = res.conversation_id
            else:
                res = mistral.beta.conversations.append(
                    conversation_id=self._current_conversation_id, inputs=txt
                )

            log.info(f"🤖 {res.usage}")
            # log.info(res)
            # for output in res.outputs:
            #     reply = getattr(output, "content", "")
            #     log.info(output)
        return res.outputs

    def chat(self):
        self.print_line("ℹ **Type in 'stop', 'bye' or 'quit' to stop chatting**")
        while True:
            self.print_line(self.hline)
            prompt = input(f"👤 {self.username} says: ")
            if prompt.lower() not in ["stop", "bye", "bye-bye", "ciao", "quit"]:
                outputs = self._conversation_tell(prompt)
                self.print_line(self.hline)
                # print(outputs)
                self.print_message_outputs(outputs)
            else:
                outputs = self._conversation_tell("I am going to stop this conversation, bye bye")
                self.print_message_outputs(outputs)
                self.print_line(self.hline)
                break

    def conversations_to_dataframe(self):
        """List all conversations from this API key"""

        with Mistral(
            api_key=self._api_key,
        ) as mistral:

            conversations_list = mistral.beta.conversations.list(page=0, page_size=100)

        if len(conversations_list) > 0:
            df = pd.DataFrame(
                [
                    (conversation.id, conversation.created_at, conversation.updated_at)
                    for conversation in conversations_list
                ],
                columns=["id", "created", "updated"],
            )
            df = df.sort_values("updated", ascending=False).reset_index(drop=True)
            return df
        else:
            return None

    def replay(self, conversation_id = None):
        """Replay a conversation

        Parameters
        ----------
        conversation_id : int, optional
            The conversation ID to replay. By default we replay the last one in the history

        See Also
        --------
        :class:`Assistant.conversations_to_dataframe`
        """
        if conversation_id is None:
            df = self.conversations_to_dataframe()
            conversation_id = df.iloc[0]["id"]

        with Mistral(
            api_key=self._api_key,
        ) as mistral:
            conversation = mistral.beta.conversations.get_messages(
                conversation_id=conversation_id
            )

            for message in conversation.messages:
                role = "👤" if message.role == 'user' else "🤖"

                if self.runner == "notebook":
                    if message.role == 'user':
                        self.print_line(self.hline)
                        self.print_line(f"<b>{message.content.capitalize()}</b>", role)
                    elif message.role == 'assistant':
                        if isinstance(message.content, str):
                            self.print_line(f"\n{message.content}", role)

                        if isinstance(message.content, list):
                            for chunk in message.content:
                                if chunk.type == 'text':
                                    self.print_line(chunk.text, role)
                                if chunk.type == 'tool_reference':
                                    if chunk.tool == 'web_search':
                                        content = f"<a href='{chunk.url}'>{chunk.title}</a><br><small>{chunk.description}</small>"
                                        self.print_source(content, "Source")

                else:
                    if message.role == 'user':
                        self.print_line("\n")
                        self.print_line(self.hline)
                        self.print_line(message.content.capitalize(), role)
                    elif message.role == 'assistant':
                        if isinstance(message.content, str):
                            self.print_line(f"\n{message.content}", role)

                        if isinstance(message.content, list):
                            for chunk in message.content:
                                if chunk.type == 'text':
                                    self.print_line(chunk.text, role)
                                if chunk.type == 'tool_reference':
                                    if chunk.tool == 'web_search':
                                        content = f"Source: {chunk.title} ({chunk.url})\n{chunk.description}"
                                        self.print_source(content, "Source")

    def clear(self):
        """Clean up data from MistralAI

        Deletes:

        - library together with all documents that have been uploaded to that library,
        -
        """
        with Mistral(
            api_key=self._api_key,
        ) as mistral:
            # Given a library id, deletes it together with all documents that have been uploaded to that library.
            res = mistral.beta.libraries.delete(library_id=self.library_id)
            log.info(res)

        return self
