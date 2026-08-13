CREATE TABLE [dbo].[RoleType] (
    [Id]         INT           IDENTITY (1, 1) NOT NULL,
    [RoleName]   NVARCHAR (50) NOT NULL,
    [RoleDescription] NVARCHAR (200) NULL,
    [IsEnable]   BIT           NULL,
    [CreatedBy]  INT           NULL,
    [CreatedOn]  DATETIME      NOT NULL,
    [ModifiedBy] INT           NULL,
    [ModifiedOn] DATETIME      NOT NULL
);
GO

