CREATE TABLE [dbo].[Position]
(
  [Id] INT NOT NULL PRIMARY KEY IDENTITY(1, 1),
  [TeamId] INT NOT NULL, 
  [PositionName] NVARCHAR(100) NOT NULL,
  [CreatedOn] DATETIME NULL,
  [CreatedBy] INT NULL,
  [ModifiedOn] DATETIME  NULL,
  [ModifiedBy] INT NULL
)
