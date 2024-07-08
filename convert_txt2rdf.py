

class ConvertTxt2RDF:

    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def read_from_file(self):
        print('High')
        outputs = []
        with open(self.input_file, "r") as f:
            lines = f.readlines()
            # print(lines)
            for line in lines:
                parts = line.replace('\n', '').replace('\'', '').split("\t")
                if len(parts) == 2:
                    tmp = f"<http://{parts[0]}>    <http://www.w3.org/2002/07/owl#sameAs>   <http://{parts[1]}>.\n"
                    outputs.append(tmp)
                else:
                    print('Error')
        self.write_to_file(outputs)

    def write_to_file(self, outputs):
        f = open(self.output_file, "w")
        for output in outputs:
            f.write(output)
        f.close()

    def convert(self):
        self.read_from_file()


if __name__ == "__main__":
    base_dir = "./inputs/"
    input_file = base_dir + "DBPedia_Actor/Actor.Goldstandard.txt"
    output_file = base_dir + "DBPedia_Actor/valid_same_as.ttl"

    convertor = ConvertTxt2RDF(input_file, output_file)
    convertor.convert()
    # ConvertTxt2RDF.write_to_file()
    # convertor.read_from_file()
