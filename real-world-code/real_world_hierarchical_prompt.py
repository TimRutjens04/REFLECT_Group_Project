import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle
import numpy as np
import random
from mdetr_object_detector import *
from real_world_scene_graph import Node as SceneGraphNode
from real_world_scene_graph import SceneGraph
from real_world_get_local_sg import get_scene_graph
from main.point_cloud_utils import *
from main.utils import *
from argparse import ArgumentParser
import zarr
import json
from imagecodecs import imread
from real_world_utils import get_robot_plan
from LLM.prompt import LLMPrompter
from AudioCLIP.real_world_audio import get_sound_events

np.random.seed(91)
random.seed(91)

llm_prompter = LLMPrompter(gpt_version="gpt-4")
device = f'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_scene_text(scene_graph):
    output = ""
    for node in scene_graph.nodes:
        node_name = node.name
        if node.state is not None:
            node_name = f"{node_name} ({node.state})"
        output += (node_name + ", ")
    if len(scene_graph.nodes) != 0:
        output = output[:-2] + ". "
    for edge in scene_graph.edges.values():
        # filter out redundant relations
        start_node_name = str(edge.start) 
        end_node_name = str(edge.end)
        if edge.edge_type == 'on the left of':
            if (end_node_name, start_node_name) in scene_graph.edges:
                other_edge = scene_graph.edges[(end_node_name, start_node_name)]
                if other_edge.edge_type == 'on the right of':
                    continue
        if edge.edge_type == 'below':
            if (end_node_name, start_node_name) in scene_graph.edges:
                other_edge = scene_graph.edges[(end_node_name, start_node_name)]
                if other_edge.edge_type == 'above':
                    continue
        if edge.edge_type == 'near':
            if (end_node_name, start_node_name) in scene_graph.edges:
                other_edge = scene_graph.edges[(end_node_name, start_node_name)]
                if other_edge.edge_type == 'on top of' or other_edge.edge_type == 'inside' or other_edge.edge_type == 'near':
                    continue
        output += (start_node_name + " is " + edge.edge_type + " " + end_node_name)
        output += ". "
    output = output[:-1]

    return output

def save_L1_images(args, task_info):
    args.folder_name = task_info["general_folder_name"]
    os.system('mkdir -p real_world/images/{}'.format(args.folder_name))
    key_frames = []
    with open('real_world/state_summary/{}/L1_key_frames.txt'.format(args.folder_name), 'r') as f:
        frames = f.readlines()
        key_frames = [int(frame) for frame in frames]
    key_frames = key_frames[1:] # remove the first frame
    for step_idx in key_frames:
        rgb = imread(f'real_world/data/{args.folder_name}/videos/0/0/color/{step_idx}.0.0.0')
        depth = imread(f'real_world/data/{args.folder_name}/videos/0/0/depth/{step_idx}.0.0')
        # depth = np.clip(depth, 0, 1000)
        # plt.matshow(depth,cmap=plt.cm.jet,interpolation='bicubic')
        # plt.axis('off')
        im = Image.fromarray(rgb)
        im.save('real_world/images/{}/rgb_{}.png'.format(args.folder_name, step_idx))
        # plt.savefig('real_world/images/{}/depth_{}.png'.format(args.folder_name, step_idx))


def run_wo_sound(args, task_info):
    args.folder_name = task_info["general_folder_name"]

    if not os.path.exists(f'real_world/state_summary/{args.folder_name}/state_summary_L2_wo_sound.txt'):
        with open('real_world/state_summary/{}/{}'.format(args.folder_name, 'state_summary_L2.txt'), 'r') as f:
            L2_captions = f.readlines()
        
        L2_captions_wo_sound = []
        for caption in L2_captions:
            if "Auditory observation" in caption:
                L2_captions_wo_sound.append(caption[:caption.find("Auditory observation")-1])
            else:
                L2_captions_wo_sound.append(caption)
        L2_summary_wo_sound = "".join(L2_captions_wo_sound)
        with open(f'real_world/state_summary/{args.folder_name}/state_summary_L2_wo_sound.txt', 'w') as f:
            f.write(L2_summary_wo_sound)
    else:
        with open(f'real_world/state_summary/{args.folder_name}/state_summary_L2_wo_sound.txt', 'r') as f:
            L2_captions_wo_sound = f.readlines()
        L2_summary_wo_sound = "".join(L2_captions_wo_sound)

    if not os.path.exists(f'real_world/state_summary/{args.folder_name}/state_summary_L1_wo_sound.txt'):
        with open('real_world/state_summary/{}/{}'.format(args.folder_name, 'state_summary_L1.txt'), 'r') as f:
            L1_captions = f.readlines()

        L1_captions_wo_sound = []
        for caption in L1_captions:
            timestep = caption.split(".")[0]
            if "Auditory observation" in caption:
                if timestep in L2_summary_wo_sound:
                    L1_captions_wo_sound.append(caption[:caption.find("Auditory observation")-1])
            else:
                L1_captions_wo_sound.append(caption)
        L1_summary_wo_sound = "".join(L1_captions_wo_sound)
        with open(f'real_world/state_summary/{args.folder_name}/state_summary_L1_wo_sound.txt', 'w') as f:
            f.write(L1_summary_wo_sound)


    with open(f'real_world/state_summary/{args.folder_name}/global_sg.pkl', 'rb') as f:
        global_sg = pickle.load(f)

    print(f"Running failure analysis...")
    run_reasoning(args=args, task=task_info, global_sg=global_sg)


def run_BLIP2(args, task):
    args.folder_name = task["general_folder_name"]
    f_name = args.folder_name
    reasoning_json_name = 'reasoning-BLIP2-direct.json'
    if not os.path.exists(f'real_world/state_summary/{args.folder_name}/{reasoning_json_name}'):
        reasoning_dict = {}

        with open('LLM/prompts-gpt4.json', 'r') as f:
            prompt_info = json.load(f)

        # get failure reason
        reason_prompt = {}
        reason_prompt['system'] = prompt_info['prompt-simple-qa-2']['template-system']
        reason_prompt['user'] = prompt_info['prompt-simple-qa-2']['template-user'].replace("[TASK_NAME]", task['name'])
        reason_prompt['user'] = reason_prompt['user'].replace("[SUCCESS_CONDITION]", task['success_condition'])

        L1_captions = []
        state_summary_L1 = ""
        with open('real_world/state_summary/{}/state_summary_L1_BLIP2.txt'.format(args.folder_name), 'r') as f:
            L1_captions = f.readlines()
        state_summary_L1 = "".join(L1_captions)
        reason_prompt['user'] = reason_prompt['user'].replace("[L1_SUMMARY]", state_summary_L1)
        reason_prompt['user'] = reason_prompt['user'].replace("[L2_SUMMARY]", get_robot_plan(args, step=None, with_obs=False))
        print(reason_prompt['user'])

        reason, _ = llm_prompter.query(prompt=reason_prompt, sampling_params=prompt_info['prompt-simple-qa-2']['params'],
                                            save=prompt_info['prompt-simple-qa-2']['save'], save_dir=f'LLM/{f_name}')
        
        # get failure steps
        step_prompt = {}
        step_prompt['system'] = prompt_info['prompt-simple-qa-step']['template-system']
        step_prompt['user'] = prompt_info['prompt-simple-qa-step']['template-user'].replace("[PREV_PROMPT]", reason_prompt['user'] + " " + reason)
        # print(step_prompt['user']) 
        step, _ = llm_prompter.query(prompt=step_prompt, sampling_params=prompt_info['prompt-simple-qa-step']['params'],
                                    save=prompt_info['prompt-simple-qa-step']['save'], save_dir=f'LLM/{f_name}')

        step_str = step.split(" ")[0]
        if step_str[-1] == '.' or step_str[-1] == ',':
            step_str = step_str[:-1]
        reasoning_dict['pred_failure_reason'] = reason
        reasoning_dict['pred_failure_step'] = step_str
        
        reasoning_dict['gt_failure_reason'] = task['gt_failure_reason']
        reasoning_dict['gt_failure_step'] = task['gt_failure_step']

        with open('real_world/state_summary/{}/{}'.format(args.folder_name, reasoning_json_name), 'w') as f:
            json.dump(reasoning_dict, f)


def LLM_direct_summary(args, task_info):
    # TODO: try L0 as well, need to pay attention to steps after ignore(s)
    args.folder_name = task_info["general_folder_name"]
    f_name = args.folder_name
    # Get L0 summary and convert all to text
    pickle_names = os.listdir(f'real_world/state_summary/{args.folder_name}/local_graphs')
    steps = sorted([int(p.split('.')[0].split("_")[-1]) for p in pickle_names])
    state_summary_L0 = ""
    with open('real_world/state_summary/{}/state_summary_L1.txt'.format(args.folder_name), 'r') as f:
        L0_captions = f.readlines()
    state_summary_L0 = "".join(L0_captions)

    # Prompt LLM
    with open('LLM/prompts-gpt4.json', 'r') as f:
        prompt_info = json.load(f)

    if not os.path.exists(f'real_world/state_summary/{args.folder_name}/summary.txt'):
        prompt = {}
        prompt['system'] = prompt_info['prompt-direct-summary']['template-system']
        prompt['user'] = prompt_info['prompt-direct-summary']['template-user'].replace("[TASK_NAME]", task['name'])
        prompt['user'] = prompt['user'].replace("[WORLD_STATE_HISTORY]", state_summary_L0)

        summary, _ = llm_prompter.query(prompt=prompt, sampling_params=prompt_info['prompt-direct-summary']['params'], 
                                    save=prompt_info['prompt-direct-summary']['save'], save_dir=f'LLM/{f_name}')
        print(summary)
        with open(f'real_world/state_summary/{args.folder_name}/summary.txt', 'w') as f:
            f.write(summary)
    else:
        with open(f'real_world/state_summary/{args.folder_name}/summary.txt', 'r') as f:
            summary = f.read()
    print(summary)
    
    reasoning_json_name = 'llm-direct-reasoning.json'
    
    if not os.path.exists(f'real_world/state_summary/{args.folder_name}/{reasoning_json_name}'):
        reasoning_dict = {}
        # Reasoning
        reason_prompt = {}
        reason_prompt['system'] = prompt_info['prompt-simple-qa']['template-system']
        reason_prompt['user'] = prompt_info['prompt-simple-qa']['template-user'].replace("[TASK_NAME]", task['name'])
        reason_prompt['user'] = reason_prompt['user'].replace("[SUMMARY]", summary)
        reason_prompt['user'] = reason_prompt['user'].replace("[SUCCESS_CONDITION]", task['success_condition'])
        # reason_prompt['user'] = reason_prompt['user'].replace("[WORLD_STATE_HISTORY]", state_summary_L0)

        reason, _ = llm_prompter.query(prompt=reason_prompt, sampling_params=prompt_info['prompt-simple-qa']['params'],
                                    save=prompt_info['prompt-simple-qa']['save'], save_dir=f'LLM/{f_name}')

        print(reason_prompt['user'], reason)
        reasoning_dict['pred_failure_reason'] = reason

        # Failure steps
        step_prompt = {}
        step_prompt['system'] = prompt_info['prompt-simple-qa-step']['template-system']
        step_prompt['user'] = prompt_info['prompt-simple-qa-step']['template-user'].replace("[PREV_PROMPT]", reason_prompt['user'] + " " + reason)
        step, _ = llm_prompter.query(prompt=step_prompt, sampling_params=prompt_info['prompt-simple-qa-step']['params'],
                                    save=prompt_info['prompt-simple-qa-step']['save'], save_dir=f'LLM/{f_name}')

        step_str = step.split(" ")[0]
        if step_str[-1] == '.' or step_str[-1] == ',':
            step_str = step_str[:-1]
        reasoning_dict['pred_failure_step'] = step_str
        
        reasoning_dict['gt_failure_reason'] = task['gt_failure_reason']
        reasoning_dict['gt_failure_step'] = task['gt_failure_step']

        with open('real_world/state_summary/{}/{}'.format(args.folder_name, reasoning_json_name), 'w') as f:
            json.dump(reasoning_dict, f)


def LLM_direct_reasoning(args, task_info):
    args.folder_name = task_info["general_folder_name"]
    f_name = args.folder_name
    # get reasoning prompt
    with open('LLM/prompts-gpt4.json', 'r') as f:
        prompt_info = json.load(f)
    
    meta_data = read_zarr(f'real_world/data/{args.folder_name}/replay_buffer.zarr')
    total_frames = len(meta_data['data/stage'])
    print("total_frames: ", total_frames)

    reasoning_json_name = 'reasoning-wo-framework.json'
    if not os.path.exists(f'real_world/state_summary/{args.folder_name}/{reasoning_json_name}'):
        reasoning_dict = {}

        # get failure reason
        reason_prompt = {}
        reason_prompt['system'] = prompt_info['prompt-simple-qa-2']['template-system']
        reason_prompt['user'] = prompt_info['prompt-simple-qa-2']['template-user'].replace("[TASK_NAME]", task['name'])
        reason_prompt['user'] = reason_prompt['user'].replace("[SUCCESS_CONDITION]", task['success_condition'])

        L1_captions = []
        state_summary_L1 = ""
        with open('real_world/state_summary/{}/state_summary_L1.txt'.format(args.folder_name), 'r') as f:
            L1_captions = f.readlines()
        state_summary_L1 = "".join(L1_captions)
        reason_prompt['user'] = reason_prompt['user'].replace("[L1_SUMMARY]", state_summary_L1)
        reason_prompt['user'] = reason_prompt['user'].replace("[L2_SUMMARY]", get_robot_plan(args, step=None, with_obs=False))
        print(reason_prompt['user'])

        reason, _ = llm_prompter.query(prompt=reason_prompt, sampling_params=prompt_info['prompt-simple-qa-2']['params'],
                                            save=prompt_info['prompt-simple-qa-2']['save'], save_dir=f'LLM/{f_name}')
        
        # get failure steps
        step_prompt = {}
        step_prompt['system'] = prompt_info['prompt-simple-qa-step']['template-system']
        step_prompt['user'] = prompt_info['prompt-simple-qa-step']['template-user'].replace("[PREV_PROMPT]", reason_prompt['user'] + " " + reason)
        # print(step_prompt['user']) 
        step, _ = llm_prompter.query(prompt=step_prompt, sampling_params=prompt_info['prompt-simple-qa-step']['params'],
                                    save=prompt_info['prompt-simple-qa-step']['save'], save_dir=f'LLM/{f_name}')

        step_str = step.split(" ")[0]
        if step_str[-1] == '.' or step_str[-1] == ',':
            step_str = step_str[:-1]
        reasoning_dict['pred_failure_reason'] = reason
        reasoning_dict['pred_failure_step'] = step_str
        
        reasoning_dict['gt_failure_reason'] = task['gt_failure_reason']
        reasoning_dict['gt_failure_step'] = task['gt_failure_step']

        with open('real_world/state_summary/{}/{}'.format(args.folder_name, reasoning_json_name), 'w') as f:
            json.dump(reasoning_dict, f)


def run_reasoning(args, task, global_sg):
    # define reasoning file name based on ablation_type
    if args.ablation_type == 0:
        if args.audio_ver == 1:
            reasoning_json_name = 'reasoning.json'
        elif args.audio_ver == 0:
            reasoning_json_name = 'reasoning-wo-sound.json'
    elif args.ablation_type == 3:
        reasoning_json_name = 'reasoning-only-L2.json'
    elif args.ablation_type == 5:
        reasoning_json_name = 'reasoning-BLIP2.json'

    if os.path.exists(f'real_world/state_summary/{args.folder_name}/{reasoning_json_name}'):
    # if False:
        with open(f'real_world/state_summary/{args.folder_name}/{reasoning_json_name}', 'r') as f:
            reasoning_dict = json.load(f)
        return
    else:
        reasoning_dict = {}

    save_dir = f'LLM/{args.folder_name}'
    os.system("mkdir -p {}".format(save_dir))

    with open('LLM/prompts-gpt4.json', 'r') as f:
        prompt_info = json.load(f)

    # Load L2 captions from state_summary_L2.txt
    if args.ablation_type == 0 and args.audio_ver == 0:
        summary_file_name = 'state_summary_L2_wo_sound.txt'
    elif args.ablation_type == 5:
        summary_file_name = 'state_summary_L2_BLIP2.txt'
    else:
        summary_file_name = 'state_summary_L2.txt'
    with open('real_world/state_summary/{}/{}'.format(args.folder_name, summary_file_name), 'r') as f:
        L2_captions = f.readlines()

    # Load L1 captions from state_summary_L1.txt
    if args.ablation_type == 0 and args.audio_ver == 0:
        summary_file_name = 'state_summary_L1_wo_sound.txt'
    elif args.ablation_type == 5:
        summary_file_name = 'state_summary_L1_BLIP2.txt'
    else:
        summary_file_name = 'state_summary_L1.txt'
    with open('real_world/state_summary/{}/{}'.format(args.folder_name, summary_file_name), 'r') as f:
        L1_captions = f.readlines()
    
    # Loop through each subgoal and check for post-condition
    print(">>> Run step-by-step subgoal-level analysis...")
    selected_caption = ""
    prompt = {}

    for caption in L2_captions:
        action = caption.split(". ")[1].split(": ")[1].lower()

        # prompt = prompt_action_binary_template.replace("[TASK_NAME]", task['name'])
        prompt['system'] = prompt_info['prompt-action-binary']['template-system']
        prompt['user'] = prompt_info['prompt-action-binary']['template-user'].replace("[ACTION]", action).replace("[OBSERVATION]", caption[caption.find("Visual observation"):])

        ans, _  = llm_prompter.query(prompt=prompt, sampling_params=prompt_info['prompt-action-binary']['params'], 
                                    save=prompt_info['prompt-action-binary']['save'], save_dir=save_dir)
        print(prompt['user'], ans)
        print("=====================================")
        is_success = int(ans.split(", ")[0] == "Yes")
        if is_success == 0:
            selected_caption = caption
            print("Failure identified at:", caption.split(".")[0])
            break

    # check corresponding L1 caption (plus previous observations) for reasoning
    if args.ablation_type == 0 or args.ablation_type == 5:
        explain_captions = L1_captions
    elif args.ablation_type == 3:
        explain_captions = L2_captions

    if len(selected_caption) != 0:
            print(">>> Get more detailed reasoning from L1...")
            step_name = selected_caption.split(".")[0]
            for _, caption in enumerate(explain_captions):
                if step_name in caption:
                    action = caption.split(". ")[1].split(": ")[1].lower()
                    prev_observations = get_robot_plan(args, step=step_name, with_obs=True)
                    if len(prev_observations) != 0:
                        prompt_name = 'prompt-action-reason'
                    else:
                        prompt_name = 'prompt-action-reason-no-history'

                    prompt['system'] = prompt_info[prompt_name]['template-system']
                    prompt['user'] = prompt_info[prompt_name]['template-user'].replace("[ACTION]", action)
                    prompt['user'] = prompt['user'].replace("[TASK_NAME]", task['name'])
                    prompt['user'] = prompt['user'].replace("[STEP]", step_name)
                    prompt['user'] = prompt['user'].replace("[SUMMARY]", prev_observations)
                    if args.ablation_type == 3:
                        prompt['user'] = prompt['user'].replace("[OBSERVATION]", caption[caption.find("Goal"):])
                    else:
                        prompt['user'] = prompt['user'].replace("[OBSERVATION]", caption[caption.find("Action"):])
                    ans, log_prob  = llm_prompter.query(prompt=prompt, sampling_params=prompt_info[prompt_name]['params'], 
                                                save=prompt_info[prompt_name]['save'], save_dir=save_dir)
                    print(prompt['user'], ans, log_prob)
                    print("=====================================")

                    reasoning_dict['pred_failure_reason'] = ans
                    # Map to L1 frames
                    if args.ablation_type == 3:
                        reasoning_dict['pred_failure_step'] = [step_name]
                    else:
                        prompt = {}
                        prompt['system'] = prompt_info['prompt-reason-step']['template-system']
                        prompt['user'] = prompt_info['prompt-reason-step']['template-user'].replace("[FAILURE_REASON]", ans)
                        time_steps, log_prob = llm_prompter.query(prompt=prompt, sampling_params=prompt_info['prompt-reason-step']['params'],
                                                                save=prompt_info['prompt-reason-step']['save'], save_dir=save_dir)
                        print("[LLM RESPONSE] time steps:", time_steps, time_steps.split(", "))
                        reasoning_dict['pred_failure_step'] = [time_step.replace(",", "") for time_step in time_steps.split(", ")]
                    break
    else:
        print(">>> All actions are executed successfully, run plan-level analysis...")
        prompt['system'] = prompt_info['prompt-plan']['template-system']
        prompt['user'] = prompt_info['prompt-plan']['template-user'].replace("[TASK_NAME]", task['name'])
        prompt['user'] = prompt['user'].replace("[SUCCESS_CONDITION]", task['success_condition'])
        prompt['user'] = prompt['user'].replace("[CURRENT_STATE]", get_scene_text(global_sg))
        prompt['user'] = prompt['user'].replace("[OBSERVATION]", get_robot_plan(args=args, step=None, with_obs=False))
        ans, _ = llm_prompter.query(prompt=prompt, sampling_params=prompt_info['prompt-plan']['params'], 
                                    save=prompt_info['prompt-plan']['save'], save_dir=save_dir)
        print(prompt['user'], ans)
        print("=====================================")
        reasoning_dict['pred_failure_reason'] = ans
        prompt['system'] = prompt_info['prompt-plan-step']['template-system']
        prompt['user'] = prompt_info['prompt-plan-step']['template-user'].replace("[PREV_PROMPT]", prompt['user'] + " " + ans)
        step, _ = llm_prompter.query(prompt=prompt, sampling_params=prompt_info['prompt-plan-step']['params'], 
                                    save=prompt_info['prompt-plan-step']['save'], save_dir=save_dir)
        print(prompt['user'], step)
        print("=====================================")
        step_str = step.split(" ")[0]
        if step_str[-1] == '.' or step_str[-1] == ',':
            step_str = step_str[:-1]
        reasoning_dict['pred_failure_step'] = step_str

    reasoning_dict['gt_failure_reason'] = task['gt_failure_reason']
    reasoning_dict['gt_failure_step'] = task['gt_failure_step']
    
    with open('real_world/state_summary/{}/{}'.format(args.folder_name, reasoning_json_name), 'w') as f:
        json.dump(reasoning_dict, f)


def generate_replan(f_name, global_sg, task, task_object_list, args):
    pass


def create_folders(f_name):
    os.system(f'mkdir -p real_world/state_summary/{f_name}')
    # os.system(f'mkdir -p real_world/state_summary/{args.folder_name}/detic_obj_det/')
    os.system(f'mkdir -p real_world/state_summary/{f_name}/mdetr_obj_det/')
    # os.system(f'mkdir -p real_world/state_summary/{args.folder_name}/detic_obj_det/images/')
    os.system(f'mkdir -p real_world/state_summary/{f_name}/mdetr_obj_det/images/')
    # os.system(f'mkdir -p real_world/state_summary/{args.folder_name}/detic_obj_det/det/')
    os.system(f'mkdir -p real_world/state_summary/{f_name}/mdetr_obj_det/det/')
    os.system(f'mkdir -p real_world/state_summary/{f_name}/mdetr_obj_det/clip_processed_det/')
    
def config_parser(parser=None):
    if parser is None:
        parser = ArgumentParser("Robot Failure Summarization")
    parser.add_argument('--tasks','--list', nargs='*')
    parser.add_argument('--folder_name', type=str, default="", help="if pipeline should be run on only one specific folder")
    parser.add_argument('--obj_det', type=str, default="mdetr", help="which object detection model to use")
    parser.add_argument('--audio_ver', type=int, default=1, help='1 is with detected audio, 0 is without audio')
    parser.add_argument('--ablation_type', type=int, default=0, help="which experiment to run")
    return parser

def read_zarr(file_path):
    meta_data = zarr.open(file_path, 'r')
    stage = np.array(meta_data['data/stage'])
    return meta_data

def get_interact_actions(meta_data, total_frames, args, task_json):
    stages = {}
    prev_stage = 0
    for step_idx in range(0, total_frames):
        curr_stage = meta_data['data/stage'][step_idx]
        # for the first frame of an action
        if curr_stage not in stages:
            stages[curr_stage] = [step_idx]
        # for the last frame of an action
        if curr_stage != prev_stage:
            # -- remove later --
            if args.folder_name == 'heatPotato1' and prev_stage == 6:
                step_idx = step_idx - 3
            if args.folder_name == 'heatPotato2' and prev_stage == 8: # 7268
                step_idx = step_idx - 2
            if args.folder_name == 'heatPotato2' and prev_stage == 9: # 10000 - 90 = 9910
                step_idx = step_idx - 90
            if args.folder_name == 'heatPotato2' and prev_stage == 10: # 11597 
                step_idx = step_idx - 4
            if args.folder_name == 'boilWater1' and prev_stage == 6: # 4685 
                step_idx = step_idx - 10
            # ------------------
            stages[prev_stage].append(step_idx-1)
        prev_stage = curr_stage
    # for the last frame for last stage
    last_idx = len(os.listdir(f'real_world/data/{args.folder_name}/videos/0/0/color'))-2
    stages[curr_stage].append(last_idx)
    actions = task_json['actions']
    interact_actions = {}
    print("stages: ", stages)
    for k, v in stages.items():
        if actions[k] == "Terminate":
            continue
        interact_actions[(v[0], v[1])] = actions[k]
    print("interaction_actions: ", interact_actions)
    return interact_actions


def run_real_world_pipeline(args, task_info):
    args.folder_name = task_info["general_folder_name"]
    create_folders(args.folder_name)
    print(f"Running reasoning for task {args.folder_name}...")

    meta_data = read_zarr(f'real_world/data/{args.folder_name}/replay_buffer.zarr')
    total_frames = len(meta_data['data/stage'])
    print("total_frames: ", total_frames)

    # Load the object detector
    if not os.path.exists(f'real_world/state_summary/{args.folder_name}/{args.obj_det}_obj_det/det/{total_frames}.pickle'):
        detector = mdetr_efficientnetB3_phrasecut(pretrained=True).to(device)
        detector.eval()
    else:
        detector = None

    # interact actions and nav actions
    interact_actions = get_interact_actions(meta_data, total_frames, args, task_info)
    print("interact_actions: ", interact_actions)
    interact_actions_start_idx = [idx[0] for idx in interact_actions.keys() if "Ignore" not in interact_actions[idx]]
    interact_actions_end_idx = [idx[1] for idx in interact_actions.keys() if "Ignore" not in interact_actions[idx]]
    print("interact_actions_end_idx: ", interact_actions_end_idx)

    ignore_dict = {}
    for key in interact_actions:
        if "Ignore" in interact_actions[key]:
            ignore_length = key[1] - key[0]
            ignore_end_idx = key[1]
            ignore_dict[ignore_end_idx] = ignore_length

    if args.audio_ver == 1:
        # get sound detection result
        audio_path= f'real_world/data/{args.folder_name}/videos/0/0/audio.wav'
        if os.path.exists(audio_path):
            volume_thresh = 0.03 if task_info['task_idx']== 3 else 0.04
            detected_sounds = get_sound_events(audio_path=audio_path, volume_thresh=volume_thresh)
        else:
            print("no audio file found")
            detected_sounds = {}
        print("detected sounds:", detected_sounds)
        
        sound_det_idx_dict = {}
        for sound_range in detected_sounds.keys():
            step_idx = sound_range[1]*30
            total_ignore_length = 0
            for ignore_end_idx in ignore_dict:
                if step_idx > ignore_end_idx:
                    total_ignore_length += ignore_dict[ignore_end_idx]
            sound_det_idx_dict[sound_range[1]*30+total_ignore_length] = detected_sounds[sound_range]
        print("sound_det_idx_dict: ", sound_det_idx_dict)
    else:
        sound_det_idx_dict = {}

    # LEVEL-0 STATE SUMMARY: Dense
    key_frames = []
    if not os.path.exists(f'real_world/state_summary/{args.folder_name}/global_sg.pkl'):
    # if True:
        os.system(f'mkdir -p real_world/state_summary/{args.folder_name}/local_graphs')
        os.system("mkdir -p real_world/scene/{}".format(args.folder_name))
        total_points_dict, bbox3d_dict = {}, {}
        prev_graph = SceneGraph()
        cnt, sr = 0, 150
        # prev_obj_held = None

        for step_idx in range(0, total_frames):
            if step_idx != 0 and step_idx not in interact_actions_end_idx and step_idx not in sound_det_idx_dict:
                continue

            print("[Frame] " + str(step_idx))
            distractor_list = []
            object_list = task_info['object_list']
            if 'distractor_list' in task_info:
                distractor_list = task_info['distractor_list']
            print("object list:", object_list)
            print("distractor list:", distractor_list)

            rgb = imread(f'real_world/data/{args.folder_name}/videos/0/0/color/{step_idx}.0.0.0')
            depth = imread(f'real_world/data/{args.folder_name}/videos/0/0/depth/{step_idx}.0.0')
            local_sg, bbox3d_dict, total_points_dict, bbox2d_dict = get_scene_graph(args, rgb, depth, step_idx, object_list, distractor_list, 
                                                                                    detector, total_points_dict, bbox3d_dict, meta_data, task_info)
            print("========================[Current Graph]=====================")
            print(local_sg)
            with open(f'real_world/state_summary/{args.folder_name}/local_graphs/local_sg_{step_idx}.pkl', 'wb') as f:
                pickle.dump(local_sg, f)

            # 1. select based on task-informed scene graph difference
            if local_sg != prev_graph:
                if step_idx not in key_frames:
                    key_frames.append(step_idx)
                    prev_graph = local_sg

            # 2. select based on interact actions
            if step_idx in interact_actions_end_idx:
                if step_idx not in key_frames:
                    key_frames.append(step_idx)

            # 3. select based on audio events
            if step_idx in sound_det_idx_dict:
                if step_idx not in key_frames:
                    key_frames.append(step_idx)
            
        # 2. Create global scene graph
        global_sg = SceneGraph()
        for label in total_points_dict.keys():
            if label in bbox3d_dict.keys() and label in bbox2d_dict.keys():
                new_node = SceneGraphNode(name=label, object_id=label, pos3d=bbox3d_dict[label].get_center(),
                        corner_pts=np.array(bbox3d_dict[label].get_box_points()), bbox2d=bbox2d_dict[label],
                        pcd=total_points_dict[label], global_node=True)
                global_sg.add_node_wo_edge(new_node)
                global_sg.add_node(new_node, rgb)
        with open(f'real_world/state_summary/{args.folder_name}/global_sg.pkl', 'wb') as f:
            pickle.dump(global_sg, f)

        # save keyframes to disk
        with open('real_world/state_summary/{}/L1_key_frames.txt'.format(args.folder_name), 'w') as f:
            for frame in key_frames:
                f.write("%i\n" % frame)
    else:
        with open(f'real_world/state_summary/{args.folder_name}/global_sg.pkl', 'rb') as f:
            global_sg = pickle.load(f)

            print("================ Global SG ================")
            print(global_sg)

    # LEVEL-1 Abstraction -- key frames
    L1_summary_file_name = 'state_summary_L1_BLIP2.txt' if args.ablation_type == 5 else 'state_summary_L1.txt'
    if not os.path.exists(f'real_world/state_summary/{args.folder_name}/{L1_summary_file_name}'):
    # if True:
        print("Generating L1 summary")
        state_summary_L1 = ""
        L1_captions = []
        key_frames = []
        with open('real_world/state_summary/{}/L1_key_frames.txt'.format(args.folder_name), 'r') as f:
            frames = f.readlines()
            key_frames = [int(frame) for frame in frames]

        for step_idx in key_frames:
            if step_idx == 0:
                continue
            caption = ""
            # add action
            for key in interact_actions:
                min_step, max_step = key
                if min_step <= (step_idx) <= max_step:
                    total_ignore_length = 0
                    for ignore_end_idx in ignore_dict:
                        if step_idx > ignore_end_idx:
                            total_ignore_length += ignore_dict[ignore_end_idx]
                    caption += f"{convert_step_to_timestep(step_idx-total_ignore_length, video_fps=30)}. Action: {interact_actions[key]}."
            
            if "Ignore" in caption or "Skip" in caption:
                continue
            
            if args.ablation_type == 5:
                with open(f'real_world/BLIP2_captions/{args.folder_name}/caption_{step_idx}.txt', 'r') as f:
                    scene_text = f.readlines()[0] + "."
            else:
                with open(f'real_world/state_summary/{args.folder_name}/local_graphs/local_sg_{step_idx}.pkl', 'rb') as f:
                    local_sg = pickle.load(f)
                    print(local_sg)
                scene_text = get_scene_text(local_sg)
            # print("scene_text: ", scene_text)
            if len(scene_text) != 0:
                caption += f" Visual observation: {scene_text}"

            if step_idx in sound_det_idx_dict:
                caption += f" Auditory observation: {sound_det_idx_dict[step_idx]}."

            caption += "\n"
            print(caption)
            if len(L1_captions) != 0 and caption.split(".")[0] == L1_captions[-1].split(".")[0]:
                continue
            state_summary_L1 += caption
            L1_captions.append(caption)
        with open(f'real_world/state_summary/{args.folder_name}/{L1_summary_file_name}', 'w') as f:
            f.write(state_summary_L1)
    else:
        print("Skip L1 summary")
        L1_captions = []
        with open(f'real_world/state_summary/{args.folder_name}/{L1_summary_file_name}', 'r') as f:
            L1_captions = f.readlines()
        state_summary_L1 = "".join(L1_captions)

    L2_summary_file_name = 'state_summary_L2_BLIP2.txt' if args.ablation_type == 5 else 'state_summary_L2.txt'
    if not os.path.exists(f'real_world/state_summary/{args.folder_name}/{L2_summary_file_name}'):
    # if True:
        print("Generating L2 summary")
        L2_captions = []
        for step_idx in interact_actions_end_idx:
            for caption in L1_captions:
                total_ignore_length = 0
                for ignore_end_idx in ignore_dict:
                    if step_idx > ignore_end_idx:
                        total_ignore_length += ignore_dict[ignore_end_idx]
                step_num = step_idx - total_ignore_length
                if convert_step_to_timestep(step_num, video_fps=30) in caption:
                    L2_captions.append(caption.replace("Action", "Goal"))

        state_summary_L2 = "".join(L2_captions)
        with open('real_world/state_summary/{}/{}'.format(args.folder_name, L2_summary_file_name), 'w') as f:
            f.write(state_summary_L2)
    else:
        print("Skip L2 summary")
        L2_captions = []
        with open('real_world/state_summary/{}/{}'.format(args.folder_name, L2_summary_file_name), 'r') as f:
            L2_captions = f.readlines()
        state_summary_L2 = "".join(L2_captions)

    # Query LLM for reasoning and replan
    print(f"Running failure analysis...")
    if args.ablation_type == 5:
        run_BLIP2(args, task_info)
    else:
        run_reasoning(args=args, task=task_info, global_sg=global_sg)


if __name__ == '__main__':
    args = config_parser().parse_args()

    task_lis = []
    # print("Args: ", args)
    task_lis = list(map(int, args.tasks))
    if task_lis == [0]:
        task_lis = list(range(1, 31))

    # f_names = []
    with open('real_world/tasks_real_world.json', 'r') as f:
        tasks_json = json.load(f)

    for task_idx in task_lis:
        task = tasks_json['Task ' + str(task_idx)]
        # four LLM baselines
        if args.ablation_type in [0, 3, 5]:
            if args.audio_ver == 0:
                run_wo_sound(args, task)
            else:
                run_real_world_pipeline(args, task)
        elif args.ablation_type == 2:
            LLM_direct_summary(args, task)
        elif args.ablation_type == 4:
            LLM_direct_reasoning(args, task)
