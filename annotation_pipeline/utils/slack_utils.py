import json
import mimetypes
from io import BytesIO

import requests


class SlackUploadError(RuntimeError):
    """Raised when any of the three upload steps fails."""


def wrap_msg_with_project_name(msg: str, projname: str) -> str:
    return f'*[{projname}]*\n{msg}'


def post_to_slack(msg: str, webhook_url: str) -> None:
    """Post a message via webhook to slack"""
    requests.post(
        webhook_url,
        data=json.dumps({'text': msg}),
        headers={'Content-Type': 'application/json'},
    )


def post_new_message_and_get_thread_id(
    text: str, slack_bot_token: str, channel_id: str
) -> str | None:
    """
    Post a new message to a Slack channel and return its thread timestamp.

    Parameters
    ----------
    text : str
        Message content to send.
    slack_bot_token : str
        Bot token with `chat:write` scope.
    channel_id : str
        Target Slack channel ID.

    Returns
    -------
    str or None
        Slack message timestamp (thread ID) if the post succeeds, otherwise
        ``None``.
    """
    url = 'https://slack.com/api/chat.postMessage'
    headers = {
        'Authorization': f'Bearer {slack_bot_token}',
        'Content-Type': 'application/json',
    }
    payload = {
        'channel': channel_id,
        'text': text,
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()

    if data.get('ok'):
        thread_ts = data['ts']
        return thread_ts
    else:
        return None


def reply_in_thread(
    text: str, thread_ts: str, slack_bot_token: str, channel_id: str
) -> None:
    """
    Reply to an existing Slack thread.

    Parameters
    ----------
    text : str
        Reply content.
    thread_ts : str
        Timestamp (thread ID) of the parent message.
    slack_bot_token : str
        Bot token with `chat:write` scope.
    channel_id : str
        Slack channel ID containing the thread.

    Returns
    -------
    None
    """
    url = 'https://slack.com/api/chat.postMessage'
    headers = {
        'Authorization': f'Bearer {slack_bot_token}',
        'Content-Type': 'application/json',
    }
    payload = {'channel': channel_id, 'text': text, 'thread_ts': thread_ts}

    requests.post(url, headers=headers, json=payload)


def upload_file_to_channel(
    buffer: BytesIO,
    filename: str,
    token: str,
    *,
    channel_id: str | None = None,
    title: str | None = None,
    initial_comment: str | None = None,
    thread_ts: str | None = None,
) -> dict:
    """
    Upload `file_bytes` to Slack using the getUploadURLExternal / completeUploadExternal
    workflow (a.k.a. Upload V2).

    Returns the JSON response of files.completeUploadExternal on success.
    Raises SlackUploadError on any failure.

    Parameters
    ----------
    file_bytes :
        The file contents.
    filename   :
        Name to give the file in Slack.
    token      :
        Bot/user token (xoxb-… or xoxp-…) with `files:write`.
    channel_id :
        Share to this channel immediately (optional).
    title      :
        Title shown in Slack (defaults to `filename`).
    initial_comment :
        Comment to accompany the file (optional).
    thread_ts :
        If provided, the file is posted as a reply to this
        parent-message timestamp (i.e. inside that thread).
        Requires `channel_id` to be set as well.
    """
    if thread_ts and not channel_id:
        raise ValueError('thread_ts requires channel_id')

    size_bytes = buffer.getbuffer().nbytes

    # Step 1 - request to upload
    r1 = requests.post(
        'https://slack.com/api/files.getUploadURLExternal',
        headers={'Authorization': f'Bearer {token}'},
        data={'filename': filename, 'length': size_bytes},
    ).json()

    if not r1.get('ok'):
        raise SlackUploadError(f'getUploadURLExternal failed: {r1["error"]}')

    upload_url = r1['upload_url']
    file_id = r1['file_id']

    # Step 2 - Upload the file
    content_type = (
        mimetypes.guess_type(filename)[0] or 'application/octet-stream'
    )
    r2 = requests.post(
        upload_url,
        files={'file': (filename, buffer, content_type)},
        timeout=30,
    )
    if not r2.ok:
        raise SlackUploadError(
            f'PUT to upload_url failed: HTTP {r2.status_code} – {r2.text[:200]}'
        )

    # Step 3 - Finalize the upload
    payload = {
        'files': [{'id': file_id, 'title': title or filename}],
    }
    if channel_id:
        payload['channel_id'] = channel_id
    if initial_comment:
        payload['initial_comment'] = initial_comment
    if thread_ts:
        payload['thread_ts'] = thread_ts

    r3 = requests.post(
        'https://slack.com/api/files.completeUploadExternal',
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
        },
        json=payload,
        timeout=10,
    ).json()

    if not r3.get('ok'):
        raise SlackUploadError(f'completeUploadExternal failed: {r3["error"]}')

    return r3  # contains 'file', 'file_id', etc.
