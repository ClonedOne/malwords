from shutil import copyfile
import os


def main():
    dir_malwords = '/home/yogaub/projects/projects_data/malrec/malwords/malwords_results'
    dir_mini = '/home/yogaub/projects/projects_data/malrec/malwords/mini_malwords'
    f_ext = '_ss.txt.gz'

    mydoom = [
        "4905aa5a-9062-4d1d-9c72-96f1bd80bf3f",
        "ecc1e3df-bdf2-43c6-962e-ad2bc2de971a",
        "f452023a-dd65-41f9-94d5-2e6bf64f1f37",
        "666803b2-3075-4808-8f75-74ec9dcec74f",
        "915c1e51-7c30-4637-b7d6-111a5ee6c3d8",
        "683261df-1863-456f-bbb5-6bd986c13882",
        "3331fbf9-e539-4090-b34c-e4fe51f228b9",
        "b2f35abd-eac6-4dcf-bf22-d20943df96b9",
        "989005dd-de35-4cb4-9e54-0d942c18cb92",
        "a906eaf4-423a-4671-ac96-9cc8e0fceb42",
        "18109a7f-492d-4818-a182-f265a39c249d",
        "9e243134-ab59-4dd9-bd89-e7d5badf5c21",
        "f4e71de6-1c75-4855-b551-8877236386a3",
        "9f9eee89-598c-4f96-86e8-dcc3a155d528",
        "ddcd3b7f-8b24-475c-b634-b3f9d02e89ac",
        "5a663e75-62eb-471f-8090-079ae3439337",
        "8a8c6569-d97e-4986-94a0-1d9cf3402574",
        "8107df7c-e2a0-4690-a9a5-c20c46bc9745",
        "16bbffd0-f6cf-4b37-8ce1-fa538d16b2d0",
        "f6c860d1-793a-43d3-a5cb-cb2a8958bd8e",
        "bdf71156-3619-41f3-ae63-be85c599685c",
        "5ac4edf3-8fd7-4f95-bb87-b5bbdb096437",
        "439b819d-2cba-4d75-b8c3-deb12fd7e901",
        "ebea3703-d4cf-433c-9f06-c32c3bf4eac7",
        "33f751d5-c767-44d4-9e6e-b57e661c8e1b",
        "d818c6a8-60bb-4123-930d-dad2a39c2447",
        "e955bbc5-6ab7-4459-b36b-61abfee3c7eb",
        "54b8cbd5-f0ca-4e53-9c6f-032923cd74af",
        "039571bd-76af-4f46-b48a-db41be59e586",
        "79ac8efe-eeea-4575-848a-c1093842ad9b",
        "e4cf053d-8b65-4b45-b716-df2511b8ce9b"
    ]

    neobar = [
        "cd4faf41-b56a-4d63-9624-ec8c62cadc59",
        "666d0908-3df1-4cc8-95db-132096abd1e8",
        "22131d80-ca12-4aad-8c31-a5706e7fe9e2",
        "48337b9c-1ea7-4f09-9b5e-c37c4bc8cbd8",
        "87a96c87-b7d2-424d-9691-5043af3227aa",
        "5f574153-dce8-49e4-bf56-e7788cdeaf74",
        "3b99d83c-1cd6-4bee-b76e-141d8077c298",
        "a5735cd1-1596-4c49-a36b-b62a9a58c9dc",
        "6a534b8c-3d0f-441c-ab79-081f2fbb6a3a",
        "d1f6d6b6-e697-43e2-b3d8-a663f338a254",
        "a8c8dd5d-ab67-491c-892d-f7ed66add3b8",
        "58238101-7ee2-4977-a479-bd81a1ce2ada",
        "22a475d6-631a-4fe4-9bc5-102ddd57f19d",
        "ed4928a6-f947-48ab-b9a3-ff25994b9eba",
        "f38bc7b5-5441-4086-bb03-f18cf8f335c6",
        "14419f29-9b31-4373-864a-2208e7065ec9",
        "34c45aac-f769-4f1d-942f-fc45cc6c0c7c",
        "930676f1-660d-4cf7-9649-8d53d325991a",
        "ac7d96b9-e253-4389-b415-f6a8a24ea25a",
        "ac6fad14-e132-4329-8175-d23fbad1fea9",
        "22aa8cea-9e52-4599-9f32-6babc8e8c48a",
        "520eec52-18ff-4d8b-b5c4-06c1161b93ad",
        "d3890f8a-23c2-4368-9772-b18032b9623a",
        "fc139498-190f-4cc0-91ce-d7fcdda92ad8",
        "cd31e5e7-cdfc-4b61-a083-752ebd9b3b7c"
    ]

    for f_name in mydoom:
        copyfile(
            os.path.join(dir_malwords, f_name + f_ext),
            os.path.join(dir_mini, f_name + f_ext)
        )

    for f_name in neobar:
        copyfile(
            os.path.join(dir_malwords, f_name + f_ext),
            os.path.join(dir_mini, f_name + f_ext)
        )

if __name__ == '__main__':
    main()
