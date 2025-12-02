import os
import pickle
import importlib
import warnings

has_ipython = (spec := importlib.util.find_spec("IPython")) is not None
if has_ipython:
    from IPython.display import display, Markdown

try:
    importlib.import_module('openai')  # noqa: E402
    import openai
except ImportError:
    pass

from ..options import OPTIONS
from .utils import path2assets


class Assistant:
    """AI Argo/Python assistant based on OpenAI chat-GPT-v3.5

    Requirements:

     - You must first sign-up to get an OpenAI API key at: https://platform.openai.com/signup?launch
     - You must install the openai python package: ``pip install --upgrade openai``

    ⚠️ Limitations
    Please note that, like any AI, the model may occasionally generate an inaccurate or imprecise answer. Always refer
    to the provided sources to verify the validity of the information given. If you find any issues with the response,
    kindly provide feedback to help improve the system.

    Examples
    --------
    import argopy
    argopy.set_options(openai_api_key='*****', user='Jane')  # https://platform.openai.com/account/api-keys

    AI = Assistant()
    AI.ask('how to load float 6903456 ?')
    AI.ask('show me how to load the Argo profile index with argopy')

    AI.chat()
    AI.replay()

    """
    name = "🤖 Medea"  # Jason's wife ! Another Argo/Jason mythology character: https://en.wikipedia.org/wiki/Medea

    @property
    def _prompt(self):
        with open(os.path.join(path2assets, "medea_def.pickle"), "rb") as f:
            p = pickle.load(f)
        return p.replace('NAME', self.name)

    @property
    def runner(self) -> str:
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return 'notebook'  # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return 'terminal'  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return 'standard'  # Probably standard Python interpreter

    def __init__(self):
        self.messages = []
        self.total_tokens = 0
        self._started = False
        if OPTIONS['openai_api_key'] is None:
            if os.getenv('OPENAI_API_KEY') is None:
                raise ValueError("You must specify a valid open-ai API key (with the argopy option 'openai_api_key' or \
environment variable 'OPENAI_API_KEY'. If you don't have an API key, you may get one here: https://platform.openai.com/account/api-keys")
            else:
                openai_api_key = os.getenv("OPENAI_API_KEY")
        else:
            openai_api_key = OPTIONS['openai_api_key']
        self._openai_api_key = openai_api_key
        openai.api_key = self._openai_api_key
        self._validate_key()

    def _validate_key(self):
        valid = False
        try:
            l = openai.Model.list()
            for model in l['data']:
                if model['id'] == 'gpt-4o': #'gpt-3.5-turbo':
                    valid = True
        except:
            warnings.warn("Something is wrong, probably your OpenAI API key ('%s') ..." % self._openai_api_key)
        return valid

    def tell(self, prompt=None):
        if prompt:
            self.messages.append({"role": "user",
                                  "content": prompt})
        chat = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=self.messages
        )
        reply = chat.choices[0].message.content
        self.messages.append({"role": "assistant",
                              "content": reply})

        self.total_tokens += chat.usage['total_tokens']

        return reply

    def print_line(self, role='', content=''):
        if self.runner == 'notebook':
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

            txt = f'{PREF}{1};{COLORS.yellow}' + role + ": " + RESET + f'{PREF}{0};{COLORS.cyan}' + content + ":" + RESET
            print(txt)

    def start(self, mute=False):
        if not self._started:
            self.username = OPTIONS['user'] if OPTIONS['user'] is not None else "You"
            if OPTIONS['mode'] == 'standard':
                user_icon = '🏊'
            elif OPTIONS['mode'] == 'expert':
                user_icon = '🏄'
            elif OPTIONS['mode'] == 'research':
                user_icon = '🚣'
            self.username = "%s %s" % (user_icon, self.username)

            self.messages = [
                {"role": "system",
                 "content": self._prompt},
                {"role": "user",
                 "content": "My name is %s" % self.username},
                {"role": "user",
                 "content": "I am an %s user in Argo data" % OPTIONS['mode']},
            ]
            reply = self.tell()
            if not mute:
                self.print_line(self.name, reply)
            self._started = True

    def chat(self):
        self.start()
        self.print_line("ℹ", "*Just type in 'stop' or 'bye' to stop chatting with me*")
        while True:
            # prompt = input('%s: ' % self.username)
            prompt = input(f'\033[1;33m' + self.username + ": " + f"\033[0m")
            if prompt.lower() not in ['stop', 'bye', 'bye-bye', 'ciao', 'quit']:
                reply = self.tell(prompt)
                self.print_line(self.name, reply)
            else:
                reply = self.tell("I am going to stop this conversation, bye bye %s" % self.name)
                self.print_line(self.name, reply)
                break

    def ask(self, question=None):
        self.start(mute=True)
        reply = self.tell(question)
        self.print_line(self.name, reply)

    def __repr__(self):
        summary = ["<argopy.Assistant>"]
        if len(self.messages) == 0:
            summary.append("You're up to start chatting or asking questions to %s, your Argo assistant" % self.name)
            summary.append("Initiate a chat session with the: chat() method")
            summary.append("or just ask a question with the: ask('text') method")
        else:
            summary.append("You already talked to %s (%i messages)" % (self.name, len(self.messages)))
            summary.append("You consumed %i tokens" % self.total_tokens)
        summary.append("Check out your API usage at: https://platform.openai.com/account/usage")
        summary.append("")
        summary.append("❗❗This is an highly experimental feature, mainly built just for fun ❗❗")
        summary.append(" ⚠️ Limitations")
        summary.append(
            "Please note that, like any AI, the model may occasionally generate an inaccurate or imprecise answer. Always refer to the Argo information sources or argopy documentation https://argopy.readthedocs.io to verify the validity of the information given. If you find any issues with the response, kindly provide feedback to help improve the system.")
        summary.append("")
        return "\n".join(summary)

    def replay(self):
        for im, m in enumerate(self.messages):
            if im > 3:
                if m['role'] == 'assistant':
                    role = self.name
                elif m['role'] == 'user':
                    role = self.username
                self.print_line(role, m['content'])
