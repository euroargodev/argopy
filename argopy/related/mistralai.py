import importlib
import os
import getpass
import pandas as pd
from typing import Literal

from ..errors import DataNotFound
from ..stores import httpstore
from ..options import OPTIONS
from .. import ArgoDocs


has_ipython = (spec := importlib.util.find_spec("IPython")) is not None
if has_ipython:
    from IPython.display import display, Markdown

try:
    importlib.import_module("mistralai")  # noqa: E402
    from mistralai import Mistral
except ImportError:
    pass


class Assistant:
    """

    With this class we experiment with the MistralAI LLMs, using the `mistralai library <https://docs.mistral.ai/getting-started/clients/>`_

    The strategy is to upload argopy and Argo manuals to an online library, and then to inform an agent with these documents (with OCR).
    A conversation with the agent can then be initiated to get reliable information about Argo.

    Notes
    -----
    Some limitations arise for free account, most notably the limitation to 20 documents.

    https://docs.mistral.ai/getting-started/quickstart/#account-setup
    https://console.mistral.ai/api-keys
    https://github.com/mistralai/client-python
    https://docs.mistral.ai/api/

    """

    def __init__(self, name : str = "argo", model : Literal["mistral-medium-2505"] = "mistral-medium-2505", **kwargs):
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

        # self.model="devstral-small-2507"
        self.model = model

    def __repr__(self):
        txt = [f"<argopy.MistralAIAgent.{self.agent_name}>"]
        txt.append(f"🔗 {self.agent_uri}")
        docs = self.info_library(errors='ignore')
        if docs is not None:
            txt.append(f"📚 '{self.library_name}' has {self.info_library().shape[0]} document(s):")
            txt.append("\n".join([f"\t- {row}" for row in docs['name']]))
        else:
            txt.append(f"No documents in the library '{self.library_name}', please use the 'fill_library()' method to upload content.")
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

    def print_line(self, role="", content=""):
        if self.runner == "notebook":
            display(Markdown("**%s**: %s" % (role, content)))
        else:
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

            txt = (
                f"{PREF}{1};{COLORS.yellow}"
                + role
                + ": "
                + RESET
                + f"{PREF}{0};{COLORS.cyan}"
                + content
                + ":"
                + RESET
            )
            print(txt)

    @property
    def library_id(self):
        if getattr(self, "_lib_id", None) is None:
            self.get_library()
        return self._lib_id

    def get_library(self):
        """Get a library ID, create if necessary
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

    def fill_library(self):
        """
        https://github.com/mistralai/client-python/blob/main/docs/sdks/documents/README.md

        Parameters
        ----------
        library_id

        Returns
        -------

        """

        list_documents = {
            "argopy-cheatsheet.pdf": "https://argopy.readthedocs.io/en/latest/_static/argopy-cheatsheet.pdf",
            # "Argo user's manual": ArgoDocs(29825).pdf.split(';')[-1],
        }
        df = ArgoDocs().list
        for ii, row in df.iterrows():
            list_documents.update({row['title']: ArgoDocs(row['id']).pdf.split(';')[-1].strip()})

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
                        print(res)

    def ls_library(self):

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

    def info_library(self, update=False, errors='raise'):
        if update or getattr(self, '_info_library', None) is None:

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
                    self._info_library = pd.DataFrame(documents, columns=["name", "number_of_pages", "size", "processing_status", "summary"])
                elif errors == 'raise':
                    raise DataNotFound("No documents found for this library.")
                else:
                    return None

        return self._info_library

    @property
    def agent_id(self):
        if getattr(self, "_agent_id", None) is None:
            self.get_agent()
        return self._agent_id

    def get_agent(self):
        """
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
                # print(res)
                print(f"{self.agent_name} created")
                self._agent_id = res.id
                self.update_agent()

        return self._agent_id

    def update_agent(self):

        with Mistral(
            api_key=self._api_key,
        ) as mistral:

            res = mistral.beta.agents.update(
                agent_id=self.agent_id,
                description="An expert agent fully aware of Argo documents",
                # instructions=prompt,
                instructions="Use the library tool to access external documents. If code is requested, just answer code without install instructions and always prefer an example based on the Argopy library wherever relevant. Assume the role of a research scientist in oceanography working on Argo floats data and programming in Python. For all your answers you will provide a reliability scale from 1 (hypothetic) to 5 (highly reliable) to indicate how confident your are in your answer.",
                tools=[
                    {"type": "document_library", "library_ids": [self.library_id]},
                    {"type": "web_search"},
                ],
                completion_args={
                    "temperature": 0.3,
                    "top_p": 0.95,
                },
            )
            # print(res)
            print(f"{self.agent_name} updated")

    @property
    def agent_uri(self):
        return f"https://console.mistral.ai/build/agents/{self.agent_id}"

    def tell(self, txt):
        reply = ""
        new = getattr(self, "_current_conversation_id", False) is False
        # self.print_line("🤖", f" This a new conversation: {new}")

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

            self.print_line("🤖", res.usage)
            for output in res.outputs:
                reply = getattr(output, "content", "")
        return reply

    def chat(self):
        self.print_line("ℹ", "*Just type in 'stop' or 'bye' to stop chatting*")

        while True:
            self.print_line("", "<hr>")
            prompt = input(f"\033[1;33m " + self.username + ": " + f"\033[0m")
            if prompt.lower() not in ["stop", "bye", "bye-bye", "ciao", "quit"]:
                reply = self.tell(prompt)
                self.print_line("", "<hr>")
                if isinstance(reply, list):
                    for chunk in reply:
                        if getattr(chunk, "text", None) is not None:
                            self.print_line("🤖", chunk.text)
                else:
                    self.print_line("🤖", reply)
            else:
                reply = self.tell("I am going to stop this conversation, bye bye")
                self.print_line("🤖", reply)
                self.print_line("", "<hr>")
                break

    def ls_conversations(self):

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

    def replay(self):
        df = self.ls_conversations()

        with Mistral(
            api_key=self._api_key,
        ) as mistral:
            conversation = mistral.beta.conversations.get_messages(
                conversation_id=df.iloc[0]["id"]
            )

        for message in conversation.messages:
            if message.type == "message.input":
                role = "<hr>👤"
                txt = f"{message.content}"
            else:
                role = "🤖"
                txt = message.content
            self.print_line(role, txt)
