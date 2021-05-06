import torch
from tqdm import tqdm

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)

        # 역전파 단계 전에, Optimizer 객체를 사용하여 (모델의 학습 가능한 가중치인)
        # 갱신할 Variable들에 대한 모든 변화도를 0으로 만듭니다. 이는 기본적으로,
        # .backward()를 호출할 때마다 변화도가 버퍼(Buffer)에 (덮어쓰지 않고) 누적되기
        # 때문입니다. 더 자세한 내용은 torch.autograd.backward에 대한 문서를 참조하세요.
        optimizer.zero_grad()  # 사용하여 수동으로 변화도 버퍼를 0으로 설정
        _, _, loss = model(**data)
        loss.backward()  # 역전파 단계: 모델의 매개변수에 대한 손실의 변화도를 계산합니다.
        optimizer.step()  # Optimizer의 step 함수를 호출하면 매개변수가 갱신됩니다.
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)

def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)

        _, _, loss = model(**data)
        final_loss += loss.item()
    return final_loss / len(data_loader)

