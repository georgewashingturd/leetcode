#include<stdio.h>
#include<stdlib.h>

typedef struct _node {
    int val;
    struct _node * next;
} node, *pnode;

void sortLinkedList(pnode head, pnode * resHead, pnode * resTail)
{
    if (!head) {
        *resHead = *resTail = NULL;
        return;
    }
    
    pnode tmp;
    pnode pivot;
    pnode lhead, rhead, ltail, rtail;
    
    lhead = ltail = rhead = rtail = NULL;
    
    pivot = head;
    tmp = pivot->next;
    
    while (tmp)
    {
        if (tmp->val <= pivot->val)
        {
            if (!lhead)
            {
                lhead = ltail = tmp;
            }
            else
            {
                ltail->next = tmp;
                ltail = tmp;
            }
        } else {
            if (!rhead)
            {
                rhead = rtail = tmp;
            }
            else
            {
                rtail->next = tmp;
                rtail = tmp;
            }
        }
        tmp = tmp->next;
    }
    
    if (ltail)
        ltail->next = NULL;
    
    if (rtail)
        rtail->next = NULL;

    
    if (lhead)
        sortLinkedList(lhead, &lhead, &ltail);
    
    if (rhead)
        sortLinkedList(rhead, &rhead, &rtail);
    
    if (lhead) {
        *resHead = lhead;
        ltail->next = pivot;
    } else {
        *resHead = pivot;
    }
        
    pivot->next = rhead;
    if (rtail) 
    {
        *resTail = rtail;
    }
    else 
    {    
        *resTail = pivot;
    }
}

void printLinkedList(pnode head)
{
    while(head)
    {
        printf("%d, ", head->val);
        head = head->next;
    }
    printf("\n");
}

int main(void)
{
    int u[] = {22,16,5,3,8,1,9,10,4,2,6,11,15,17,12,7};
    
    pnode head, tail;
    head = NULL;
    
    for (int i = 0; i < sizeof(u)/sizeof(int); i++)
    {
        pnode tmp = (pnode)malloc(sizeof(node));
        tmp->val = u[i];
        
        if (!head){
            head = tmp;
            tail = tmp;
        } 
        else 
        {
            tail->next = tmp;
            tail = tmp;
        }
    }
    tail->next = NULL;
    
    printLinkedList(head);
    
    sortLinkedList(head,&head,&tail);
    
    printLinkedList(head);
}