CREATE TABLE [dbo].[JobGrade] (
    [Id] INT IDENTITY (1, 1) NOT NULL,
    [PositionId]   INT NOT NULL,
    [JobGradeName] NVARCHAR (50) NOT NULL, 
    [CreatedBy]    INT      NULL,
    [CreatedOn]    DATETIME NULL,
    [ModifiedBy]   INT      NULL,
    [ModifiedOn]   DATETIME NULL
);