import warnings
import importlib
import inspect
from functools import wraps

warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)


def refactored(func1):

    rel = importlib.import_module('argopy.related')
    utils = importlib.import_module('argopy.utils')
    in_related = hasattr(rel, func1.__name__)
    func2 = getattr(rel, func1.__name__) if in_related else getattr(utils, func1.__name__)

    func1_type = 'function'
    if inspect.isclass(func1):
        func1_type = 'class'

    func2_loc = 'utils'
    if in_related:
        func2_loc = 'related'

    msg = "The 'argopy.utilities.{name}' {ftype} has moved to 'argopy.{where}.{name}'. \
You're seeing this message because you called '{name}' imported from 'argopy.utilities'. \
Please update your script to import '{name}' from 'argopy.{where}'. \
After 0.1.15, importing 'utilities' will raise an error."

    @wraps(func1)
    def decorator(*args, **kwargs):
        # warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(
            msg.format(name=func1.__name__, ftype=func1_type, where=func2_loc),
            category=DeprecationWarning,
            stacklevel=2
        )
        # warnings.simplefilter('default', DeprecationWarning)
        return func2(*args, **kwargs)

    return decorator

# Argo related dataset and Meta-data fetchers

@refactored
class TopoFetcher:
    pass

@refactored
class ArgoDocs:
    pass

@refactored
class ArgoNVSReferenceTables:
    pass

@refactored
class OceanOPSDeployments:
    pass

@refactored
def get_coriolis_profile_id(*args, **kwargs):
    pass

@refactored
def get_ea_profile_page(*args, **kwargs):
    pass

@refactored
def load_dict(*args, **kwargs):
    pass

@refactored
def mapp_dict(*args, **kwargs):
    pass

# Checkers
@refactored
def is_box(*args, **kwargs):
    pass

@refactored
def is_indexbox(*args, **kwargs):
    pass

@refactored
def is_list_of_strings(*args, **kwargs):
    pass

@refactored
def is_list_of_dicts(*args, **kwargs):
    pass

@refactored
def is_list_of_datasets(*args, **kwargs):
    pass

@refactored
def is_list_equal(*args, **kwargs):
    pass

@refactored
def check_wmo(*args, **kwargs):
    pass

@refactored
def is_wmo(*args, **kwargs):
    pass

@refactored
def check_cyc(*args, **kwargs):
    pass

@refactored
def is_cyc(*args, **kwargs):
    pass

@refactored
def check_index_cols(*args, **kwargs):
    pass

@refactored
def check_gdac_path(*args, **kwargs):
    pass

@refactored
def isconnected(*args, **kwargs):
    pass

@refactored
def isalive(*args, **kwargs):
    pass

@refactored
def isAPIconnected(*args, **kwargs):
    pass

@refactored
def erddap_ds_exists(*args, **kwargs):
    pass

@refactored
def urlhaskeyword(*args, **kwargs):
    pass


# Data type casting:

@refactored
def to_list(*args, **kwargs):
    pass

@refactored
def cast_Argo_variable_type(*args, **kwargs):
    pass

from .utils.casting import DATA_TYPES

# Decorators

@refactored
def deprecated(*args, **kwargs):
    pass

@refactored
def doc_inherit(*args, **kwargs):
    pass

# Lists:

@refactored
def list_available_data_src(*args, **kwargs):
    pass

@refactored
def list_available_index_src(*args, **kwargs):
    pass

@refactored
def list_standard_variables(*args, **kwargs):
    pass

@refactored
def list_multiprofile_file_variables(*args, **kwargs):
    pass

# Cache management:
@refactored
def clear_cache(*args, **kwargs):
    pass

@refactored
def lscache(*args, **kwargs):
    pass

# Computation and performances:
@refactored
class Chunker:
    pass

# Accessories classes (specific objects):
@refactored
class float_wmo:
    pass

@refactored
class Registry:
    pass

# Locals (environments, versions, systems):
@refactored
def get_sys_info(*args, **kwargs):
    pass

@refactored
def netcdf_and_hdf5_versions(*args, **kwargs):
    pass

@refactored
def show_versions(*args, **kwargs):
    pass

@refactored
def show_options(*args, **kwargs):
    pass

@refactored
def modified_environ(*args, **kwargs):
    pass


# Monitors
@refactored
def badge(*args, **kwargs):
    pass

@refactored
class fetch_status:
    pass

@refactored
class monitor_status:
    pass

# Geo (space/time data utilities)
@refactored
def toYearFraction(*args, **kwargs):
    pass

@refactored
def YearFraction_to_datetime(*args, **kwargs):
    pass

@refactored
def wrap_longitude(*args, **kwargs):
    pass

@refactored
def wmo2box(*args, **kwargs):
    pass

# Computation with datasets:
@refactored
def linear_interpolation_remap(*args, **kwargs):
    pass

@refactored
def groupby_remap(*args, **kwargs):
    pass

# Manipulate datasets:
@refactored
def drop_variables_not_in_all_datasets(*args, **kwargs):
    pass

@refactored
def fill_variables_not_in_all_datasets(*args, **kwargs):
    pass

# Formatters:
@refactored
def format_oneline(*args, **kwargs):
    pass

@refactored
def argo_split_path(*args, **kwargs):
    pass


# Loggers
@refactored
def warnUnless(*args, **kwargs):
    pass

@refactored
def log_argopy_callerstack(*args, **kwargs):
    pass

if __name__ == "argopy.utilities":
    warnings.warn(
        "The 'argopy.utilities' has moved to 'argopy.utils'. After 0.1.15, importing 'utilities' "
        "will raise an error. Please update your script.",
        category=DeprecationWarning,
        stacklevel=2,
    )


import os
import pickle

try:
    importlib.import_module('openai')  # noqa: E402
    import openai
except ImportError:
    pass
    
    
class Assistant:
    """AI Argo/Python assistant based on OpenAI chat-GPT-v3.5

    Requirements:

     - You must first sign-up to get an OpenAI API key at: https://platform.openai.com/signup?launch
     - You must install the openai python package: ``pip install --upgrade openai``

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
        with open(os.path.join(path2pkl, "medea_def.pickle"), "rb") as f:
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
                if model['id'] == 'gpt-3.5-turbo':
                    valid = True
        except:
            warnings.warn("Something is wrong, probably your OpenAI API key ('%s') ..." % self._openai_api_key)
        return valid

    def tell(self, prompt=None):
        if prompt:
            self.messages.append({"role": "user",
                                  "content": prompt})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
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
        summary.append("Use at your own risk and be aware that chatGPT often tends to invent non-existing argopy methods")
        summary.append("The argopy documentation is the most reliable source of information: https://argopy.readthedocs.io")
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