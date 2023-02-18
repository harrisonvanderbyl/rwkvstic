# command = python3 -m rwkvstic --server **args

import torch
from sys import argv


def fixDtype(x): return torch.float32 if x == "float32" else torch.float64 if x == "float64" else torch.bfloat16 if x == "bfloat16" else torch.float16 if x == "float16" else x
def fixNumbers(x): return int(x) if type(x) == str and x.isnumeric() else x


def fixBool(x): return True if type(x) == str and x.lower(
) == "true" else False if type(x) == str and x.lower() == "false" else x


args = argv[2:]
print(args)
args = [arg.split("=") for arg in args]
print(args)

args = [[arg[0], fixDtype(arg[1])] for arg in args]
print(args)
args = [[arg[0], fixNumbers(arg[1])] for arg in args]
print(args)
args = [[arg[0], fixBool(arg[1])] for arg in args]
print(args)

args = {arg[0]: arg[1] for arg in args}
print(args)

# fix dtype and runtimedtypeinto torch types


def runServer():
    from rwkvstic.load import RWKV

    model = RWKV(**args)

    import inquirer

    # Create server using http
    PORT = args.get("port", None)

    if PORT is None:
        PORT = inquirer.prompt([inquirer.Text("port", message="What port do you want to use?")])[
            "port"]

    import http.server
    import socketserver

    # create a custom handler

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(bytes("Hello World!", "utf-8"))

        def do_POST(self):
            # load post body as json file
            import json

            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            body = json.loads(body)
            if body.get("state", None) is not None:
                body["state"] = model.initTensor(body["state"])

            if body.get("input", None) is not None:
                body["state"] = model.loadContext(
                    newctx=body["input"], statex=body.get("state", None))[1]
                del body["input"]

            print(body)
            output = model.forward(**body)
            output["state"] = [x.cpu().numpy().tolist()
                               for x in output["state"]]
            output["logits"] = output["logits"].cpu().numpy().tolist()

            self.send_response(200)
            self.send_header("Content-type", "text/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps(output), "utf-8"))

    with socketserver.TCPServer(("", int(PORT)), Handler) as httpd:
        print("serving at port", PORT)
        httpd.serve_forever()
