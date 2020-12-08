# system
from datetime import datetime
# Discord
import operator
import asyncio
# words
from textblob import TextBlob
from collections import namedtuple
# math
import re
import math
# nltk.download('punkt')
# nltk.download('vader_lexicon')
# redbot
import discord
from redbot.core.utils import AsyncIter
from redbot.core import Config
from redbot.core.utils.chat_formatting import box
from redbot.core import commands

# Dolos
import dolos.dodgyness as dodgyness
from dolos import corpus
import dolos.regression as regression
from dolos.article_parser.article_parser import ArticleParser

link_regex = re.compile(
    '((https?):((//)|(\\\\))+([\d:#@%/;$()~_?+-=\\\.&](#!)?)*)', re.DOTALL)
#\w
"""A custom cog to query URLs"""

'''
This spaghetti code was proudly thrown together during a rushed semester

It combines existing functionality from several 'Cogs' (add-ons) for the Bot framework `Redbot` 
that we have utilised.


- Message React functionality: https://github.com/flapjax/FlapJack-Cogs/tree/red-v3-rewrites/msgvote : MIT License
- Leaderboard cog:
- msgvote

'''

PlaceMarker = namedtuple("PlaceMarker", "text link")
i = 0


class dolos(commands.Cog):
    print("\n New Run \n")

    def __init__(self, bot):
        """Initialise"""
        self.bot = bot
        #self.config = Config.get_conf(self, identifier=82466655781)
        #self.config.register_user()

        self.config = Config.get_conf(self, identifier=48448948948948)
        self.config.register_user(queries=[], points=0, mention=True, next_dunce=0)
        self.config.register_guild(query=":question:")

        # self.conf = Config.get_conf(self, identifier=UNIQUE_ID, force_registration=True)

        default_guild = {
            "bot_react": False,
            "duration": 300,
            "threshold": 3,  # threshold for comment removal
            "up_emoji": "👍",
            "dn_emoji": "👎",
            "warn_emoji": "⚠️",
            "query_emoji": "❓",
        }

        super(dolos, self).__init__()  # super().__init__(*args, **kwargs)
        self.config.register_guild(**default_guild)

    @commands.group()
    async def dolos(self, ctx):
        """
        Welcome to Dolos! The subcommands available are listed below.

        These are *subcommands* and should be entered in the

        `?dolos <subcommand>` format
        """


    '''
    Commands 
    '''

    @dolos.command(name="q")
    async def query(self, ctx, message):
        """
        Ask Dolos to query a URL
        """
        try:
            '''
            Parse the text with `newspaper`
            '''
            content = ctx.message.clean_content
            print(content)
            # article_parser = ArticleParser(message.content)
            # data = article_parser.parse()

            await ctx.channel.send("Performing NLP...",
                                       allowed_mentions=discord.AllowedMentions(
                                           everyone=False, roles=False, users=False))
        except (discord.HTTPException, discord.Forbidden,):
            pass

    @dolos.command(name="ask")
    async def ask(self, ctx, q: str):
        """
        Machine Learning on text (Experimental!)

        Currently hardcoded with the question `"What is the coronavirus?"`

        BERT (Bidirectional Encoder Representations from Transformers) is designed to pre-train deep bidirectional representations from unlabeled text.

        The pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.
        """
        await ctx.send("Parsing question (this may take awhile)....")
        question = "What is the coronavirus?"
        response = dodgyness.nlpQuestion(question)

        print("res", response)

        await ctx.send(response)
        #dn_emoji, up_emoji = await self.reaction_set_1(ctx.send)



    @dolos.command(name="board")
    async def board(self, ctx, page_list: int = 1):
        """
        Show the leaderboard.
        """
        users = []
        title = "Global Dunce Leaderboard for {}\n".format(
            self.bot.user.name)
        all_users = await self.config.all_users()
        if str(all_users) == "{}":
            await ctx.send("The leaderboard is empty... Nobody's popular, for now.")
            return
        for user_id in all_users:
            user_name = await self._get_user_name(user_id)
            users.append((user_name, all_users[user_id]["points"]))
            if ctx.author.id == user_id:
                user_stat = all_users[user_id]["points"]

        board_type = "Rep"
        icon_url = self.bot.user.avatar_url
        sorted_list = sorted(
            users, key=operator.itemgetter(1), reverse=True)
        rank = 1
        for allusers in sorted_list:
            if ctx.author.name == allusers[0]:
                author_rank = rank
                break
            rank += 1
        footer_text = "Your Rank: {}                      {}: {}".format(
            author_rank, board_type, user_stat
        )

        # multiple page support
        page = 1
        per_page = 15
        pages = math.ceil(len(sorted_list) / per_page)
        if 1 <= page_list <= pages:
            page = page_list
        else:
            await ctx.send("**Please enter a valid page number! (1 - {})**".format(str(pages)))
            return

        msg = ""
        msg += "Rank     Name                   (Page {}/{})     \n\n".format(
            page, pages)
        rank = 1 + per_page * (page - 1)
        start_index = per_page * page - per_page
        end_index = per_page * page

        default_label = "  "
        special_labels = ["♔", "♕", "♖", "♗", "♘", "♙"]

        async for single_user in AsyncIter(sorted_list[start_index:end_index]):
            if rank - 1 < len(special_labels):
                label = special_labels[rank - 1]
            else:
                label = default_label
            print(single_user[0])
            msg += "{:<2}{:<2}{:<2} # {:<15}".format(
                rank, label, "➤", await self._truncate_text(single_user[0], 15)
            )
            msg += "{:>5}{:<2}{:<2}{:<5}\n".format(
                " ", " ", " ", " {}: ".format(
                    board_type) + str(single_user[1])
            )
            rank += 1
        msg += "--------------------------------------------            \n"
        msg += "{}".format(footer_text)

        em = discord.Embed(description="", colour=await self.bot.get_embed_colour(ctx))
        em.set_author(name=title, icon_url=icon_url)
        em.description = box(msg)

        await ctx.send(embed=em)

    @dolos.command(name="mention")
    async def mention(self, ctx: commands.Context, mention: bool):
        """
        Choose if I mention someone when I give them a dunce point
        ?dolos mention false/true
        """
        await self.config.user(ctx.author).mention.set(mention)
        await ctx.send("Mention setting set to `{bool}`".format(bool=mention))

    @dolos.command(name="threshold")
    async def _dolos_threshold(self, ctx, threshold: int):
        """
        Set the threshold for msg deletion. 0 = Disabled
        """

        if threshold < 0:
            await ctx.send("Invalid threshold. Must be a positive integer, or 0 to disable.")
        elif threshold == 0:
            await self.config.guild(ctx.guild).threshold.set(threshold)
            await ctx.send("Message deletion disabled.")
        else:
            await self.config.guild(ctx.guild).threshold.set(threshold)
            await ctx.send(
                "Messages will be deleted if [downvotes - "
                "upvotes] reaches {}.".format(threshold)
            )

    @commands.Cog.listener()
    async def on_message_without_command(self, message):
        """Handle on_message."""
        if not isinstance(message.channel, discord.TextChannel):
            # this is a DM or group DM, discard early
            return

        if message.type != discord.MessageType.default:
            # this is a system message, discard early
            return

        if message.author.id == self.bot.user.id:
            # this is ours, discard early
            return

        if message.author.bot:
            # this is a bot, discard early
            return
        content = message.clean_content
        if len(content) == 0:
            # nothing to do, exit early
            return
        # if content:

    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        """Handle adding warns"""

        await self._handle_reaction(payload, True)

    async def _handle_reaction(self, payload: discord.RawReactionActionEvent, add: bool):

        # get emoji
        try:
            query = await self.config.guild(self.bot.get_guild(payload.guild_id)).query_emoji()
            warn = await self.config.guild(self.bot.get_guild(payload.guild_id)).warn_emoji()
        except AttributeError:
            # Ignore out-of-guild payloads
            print("Ignore out-of-guild payloads")
            pass

        # Ignore reactions that are not query
        if payload.emoji.name == warn:
            message = await self.bot.get_channel(payload.channel_id).fetch_message(payload.message_id)
            await asyncio.sleep(5)
            await message.channel.send("Please give reasoning for disputing this call which will then be voted on.",
                                       allowed_mentions=discord.AllowedMentions(
                                           everyone=False, roles=False, users=False))
            await asyncio.sleep(15)

        if payload.emoji.name != query:
            return
        user = self.bot.get_user(payload.user_id)
        queries = await self.config.user(user).queries()
        if add:
            message = await self.bot.get_channel(payload.channel_id).fetch_message(payload.message_id)
            url = message.clean_content
            if not url:
                try:
                    url = message.attachments[0].filename
                except IndexError:
                    try:
                        url = message.embeds[0].title
                    except IndexError:
                        url = message.system_content[:50]
            queries.append(PlaceMarker(url or "[no url]",
                                       message.jump_url))
        else:
            for i, mark in enumerate(queries):
                # Unpack jump_url from PlaceMarker
                print(mark)
                _, link = mark
                if link == f"https://discordapp.com/channels/{payload.guild_id}/{payload.channel_id}/{payload.message_id}":
                    del queries[i]
                    break

        await self.method_name(message, queries, url, user)

    async def method_name(self, message, queries, url, user):

        # Add emojis
        dn_emoji, up_emoji = await self.reaction_set_1(message)
        article_parser = ArticleParser(url)
        data = article_parser.parse()
        await message.channel.send("Parsing the article for sentiment...",
                                   allowed_mentions=discord.AllowedMentions(
                                       everyone=False, roles=False, users=False))
        article_info, out_str = await self.nlp(data, message)
        await self.make_graphics(article_info, message)
        '''
        Corpus Checks
        '''
        # guard.csv
        string = corpus.analyse(url)
        # guard2.csv
        string2 = corpus.analyse(url)
        # sourcesUncut.csv
        # string3 = dodgyness.compareArticleToOtherSites(url, False)
        if string or string2:
            await self.assign_points(message, string)
        else:
            await asyncio.sleep(3)
            await message.channel.send("Domain doesn't appear on our blacklists",
                                       allowed_mentions=discord.AllowedMentions(
                                           everyone=False, roles=False, users=False))
            await asyncio.sleep(3)
            await message.channel.send("Performing Logical Regression...",
                                       allowed_mentions=discord.AllowedMentions(
                                           everyone=False, roles=False, users=False))
            reg = regression.main(data)
            await asyncio.sleep(3)
            if out_str[1] < 0.01:
                reg = 0.1
            string_builder = '~{:.0f}\\% chance this article is fake news'.format(
                reg * 100)

            await message.channel.send(string_builder,
                                       allowed_mentions=discord.AllowedMentions(
                                           everyone=False, roles=False, users=False))

            warn_emoji = await self.config.guild(message.guild).warn_emoji()

            async for message in message.channel.history(
                    limit=1, oldest_first=False  # before=before, after=after
            ):
                await message.add_reaction(up_emoji)
                await message.add_reaction(dn_emoji)
                await message.add_reaction(warn_emoji)

                print(message)
            '''
            ml-unused
            '''
            # question = "What is the coronavirus?"
            # response = dodgyness.nlpQuestion(question)
        await self.config.user(user).queries.set(queries)

    async def assign_points(self, message, string):
        await message.channel.send(string,
                                   allowed_mentions=discord.AllowedMentions(
                                       everyone=False, roles=False, users=False))

        user_points = await self.config.user(message.author).points()
        await self.config.user(message.author).points.set(user_points + 1)
        await message.channel.send(
            "**{receiver} received a dunce point!\nYou now"
            " have {reps} dunce point{plural}. \n Type `?dolos board` to see the loserboard**".format(
                receiver=await self._user_mention(message.author),
                author=message.author,
                reps=user_points + 1,
                plural="s" if (user_points + 1) > 1 else "",
            )
        )
        # return

    async def make_graphics(self, article_info, message):
        '''
        Chart
        '''
        # chart = dodgyness.getEmotion2(data, plotAttitude=True)
        # chart = dodgyness.makeGraph2(url)
        # Prints the chart
        chart = dodgyness.makeGraph(article_info)
        await asyncio.sleep(1.5)
        await message.channel.send(file=discord.File(chart, "chart.png"))
        await asyncio.sleep(1.5)
        # chart = dodgyness.makePlot(articleInfo)
        # await message.channel.send(file=discord.File(chart, "chart2.png"))

    async def nlp(self, data, message):
        '''
        Get the sentiment with `TextBlob`
        '''
        # getEmotion()
        article_info = dodgyness.getEmotion(data, plotAttitude=True)
        # TextBlob sentiment
        blob = TextBlob(str(data['sentences']))
        article_info['polarity'] = blob.sentiment.polarity
        article_info['sentiment'] = blob.sentiment.subjectivity
        out_str = 'Emotional Charge: ', article_info['emotional charge'], 'attitude:', article_info[
            'attitude'], 'polarity:', article_info['polarity'], 'sentiment:', article_info['sentiment']
        print("out", out_str)
        out_string_builder = 'Emotional Charge: {} \n Attitude: {} \n Polarity: {} \n Sentiment: {} \n  '.format(
            out_str[1], out_str[3], out_str[5], out_str[7])
        await asyncio.sleep(2)
        await message.channel.send(out_string_builder,
                                   allowed_mentions=discord.AllowedMentions(
                                       everyone=False, roles=False, users=False))
        return article_info, out_str

    async def reaction_set_1(self, message):
        up_emoji = await self.config.guild(message.guild).up_emoji()
        dn_emoji = await self.config.guild(message.guild).dn_emoji()
        query_emoji = await self.config.guild(message.guild).query_emoji()
        await message.add_reaction(up_emoji)
        await asyncio.sleep(0.5)
        await message.add_reaction(dn_emoji)
        return dn_emoji, up_emoji

    @dolos.command()
    async def flagged(self, ctx: commands.Context):
        """View links you've queried"""
        flagged = await self.config.user(ctx.message.author).queries()
        try:
            embed_permission = ctx.message.channel.permissions_for(ctx.message.guild.me).embed_links
        except AttributeError:
            # Probably means we're in a DM
            embed_permission = True
        if embed_permission:
            payload = ""
            for preview, link in flagged:
                payload += f"[{preview}]({link})\n"
            embed = discord.Embed(title="Flagged", description=payload)
            await ctx.send(embed=embed)
        else:
            payload = "**Flagged**"
            for preview, link in flagged:
                payload += f"\n[{preview}]({link})"
            await ctx.send(payload)

    @commands.Cog.listener()
    async def on_reaction_remove(self, reaction, user):
        if user.id == self.bot.user.id:
            return
        await self.count_votes(reaction)

    '''
    Static Methods
    '''

    async def count_votes(self, reaction):
        message = reaction.message
        if not message.guild:
            return
        if not reaction.me:
            return
        if await self.config.guild(message.guild).threshold() == 0:
            return
        up_emoji = self.fix_custom_emoji(await self.config.guild(message.guild).up_emoji())
        dn_emoji = self.fix_custom_emoji(await self.config.guild(message.guild).dn_emoji())
        if reaction.emoji not in (up_emoji, dn_emoji):
            return
        age = (datetime.utcnow() - message.created_at).total_seconds()
        if age > await self.config.guild(message.guild).duration():
            return
        # We have a valid vote so we can count the votes now
        upvotes = 0
        dnvotes = 0
        for react in message.reactions:
            if react.emoji == up_emoji:
                upvotes = react.count
            elif react.emoji == dn_emoji:
                dnvotes = react.count
        if (dnvotes - upvotes) >= await self.config.guild(message.guild).threshold():
            try:
                await message.delete()
            except discord.errors.Forbidden:
                await message.channel.send(
                    "I require the 'Manage Messages' permission to delete downvoted messages!"
                )

    def fix_custom_emoji(self, emoji):
        if emoji[:2] != "<:":
            return emoji
        for guild in self.bot.guilds:
            for e in guild.emojis:
                if str(e.id) == emoji.split(":")[2][:-1]:
                    return e
        return None

    @staticmethod
    async def _truncate_text(text, max_length):
        if len(text) > max_length:
            return text[: max_length - 1] + "…"
        return text

    async def _get_user_name(self, user_id: int):
        user = self.bot.get_user(user_id)
        if user is None:  # User not found
            try:
                user = await self.bot.fetch_user(user_id)
            except (discord.NotFound, discord.HTTPException):
                return "Unknown User"
        return user.name

    async def _give_rep(
            self, ctx: commands.Context, user: discord.User, current_time_as_seconds: int
    ):
        user_points = await self.config.user(user).points()
        await self.config.user(user).points.set(user_points + 1)

        await ctx.send(
            "**{receiver} received a dunce point from {author}!\nYou now"
            " have {reps} dunce point{plural}.**".format(
                receiver=await self._user_mention(user),
                author=ctx.author,
                reps=user_points + 1,
                plural="s" if (user_points + 1) > 1 else "",
            )
        )
        await self.config.user(ctx.author).next_dunce.set(current_time_as_seconds)

    async def _user_mention(self, user):
        if await self.config.user(user).mention():
            return user.mention
        return user
