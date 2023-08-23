# API call to GPT-4

import os
import openai
import time
import tiktoken
import sys
import colorama
import pyperclip

# set API keys
openai.api_key = os.getenv("OPENAI_API_KEY")

# set names of files to save conversations and token counts and costs to
convo_path = "conversations"
if not os.path.exists(convo_path):
    os.makedirs(convo_path)

# set the colors for the conversation
colorama.init()
mc = 36 # message color
sc = 35 # system color
wc = 33 # warning color

# ~~~~~~~~~~~~~~~~~~~~~~~ CLASSES ~~~~~~~~~~~~~~~~~~~~~~~#
class Conversation:
    """A conversation with the GPT engine.
       A conversation is a list of interactions."""
    
    def __init__(self, system_msg) -> None:
        # set the conversation file based on the time the conversation started
        self.convo_file = f"{convo_path}/{get_time('file')}"

        # set default settings
        self.continuing = False
        self.model = "gpt-3.5-turbo" # gpt-4
        self.max_tokens = 1000
        self.temperature = .5

        # set the conversation to an empty list
        self.messages = [{"role": "system", "content": system_msg}]

        # set the token count to 0
        self.token_count = 0
    
    def get_response(self):
        """Get a response from the GPT engine."""
        response = openai.ChatCompletion.create(
            model = self.model,
            messages = self.messages,
            temperature = self.temperature,
            max_tokens = self.max_tokens - self.token_count
        )

        # get the response text
        response_text = response["choices"][0]["text"]

        # update token count
        self.token_count += response["usage"]["total_tokens"]

        return response
    
    def stream_response(self):
        """Stream a response from the GPT engine."""
        response_text = ""

        for chunk in openai.ChatCompletion.create(
            model = self.model,
            messages = self.messages,
            temperature = self.temperature,
            max_tokens = self.max_tokens - self.token_count,
            stream = True
        ):
            content_response = chunk["choices"][0]["delta"]
            content = content_response.get('content', '')
            if content is not None:
                 response_text += content
                 print(content, flush=True, end='')
        print('\n')

        # update token count
        enc = tiktoken.encoding_for_model(self.model)
        self.token_count += len(enc.encode(response_text))
        self.token_count += len(enc.encode(self.messages[-1]["content"]))

        return response_text
    
# ~~~~~~~~~~~~~~~~~~~~~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~#
def get_time(type = "text"):
    '''Get the current time in the format "dd/mm/yyyy hh:mm:ss" for text or "yyyymmdd_hhmmss" for file names.'''
    if type == "text":
        return time.strftime("%d/%m/%Y %H:%M:%S", time.localtime())
    elif type == "file":
        return time.strftime("%Y%m%d_%H%M%S", time.localtime())
    else:
        raise ValueError("type must be either 'text' or 'file'")

def colf(message, color):
    """Formats a message in the message color."""
    return('\033[' + str(color) + 'm' + message + '\033[0m')

def get_prompt(whom = "You", col = mc):
    """Get the next user prompt for the conversation."""
    message = input(colf(whom + ": ", col))
    return check_for_clipboard(message)

def check_for_clipboard(prompt):
    '''Replace {clipboard} or {clip} with the contents of the clipboard'''
    if prompt == "clip":
        prompt = pyperclip.paste()
    else:
        for word in ["{clipboard}", "{clip}"]:
            if word in prompt:
                prompt = prompt.replace(word, pyperclip.paste())
        # Pastes the clipboard in a "code block" (triple backticks)
        if "{cn}" in prompt:
            prompt = prompt.replace("{cn}", f"\n\n'''\n{pyperclip.paste()}\n'''")
    return prompt

# ~~~~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~~#
def main():
    """Main function."""
    # ask for a system message
    system_msg = get_prompt("System message", sc)

    while True:
        # create a new conversation
        C = Conversation(system_msg)
        print(colf(f"Conversation started at {get_time()}", sc))
        
        try:
            # loop until the conversation is over
            while True:
                # Ask for a message and add it to the conversation
                message = get_prompt()
                C.messages.append({"role": "user", "content": message})

                # stream the response
                print(colf("Bot: ", mc), end='')
                response = C.stream_response()

                if not C.continuing:
                    break

        except Exception as e:
            print(e)

            # Print the line the error occurs on 
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        except KeyboardInterrupt:
            sys.exit()

if __name__ == "__main__":
    main()