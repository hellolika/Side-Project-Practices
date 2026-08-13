CREATE TYPE [dbo].[SlackUserDataType] AS TABLE
(
    [Id] Nvarchar(100) NOT NULL PRIMARY KEY,
    [Username] NVARCHAR(100),
    [RealName] NVARCHAR(100),
    [Email] NVARCHAR(255)
)