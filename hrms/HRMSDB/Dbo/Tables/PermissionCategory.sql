CREATE TABLE [dbo].[PermissionCategory] (
    [Id]                     INT           IDENTITY (1, 1) NOT NULL,
    [PermissionCategoryName] NVARCHAR (50) NOT NULL,
    [IsEnable]               BIT           NULL,
    [CreatedBy]              INT           NULL,
    [CreatedOn]              DATETIME      NOT NULL,
    [ModifiedBy]             INT           NULL,
    [ModifiedOn]             DATETIME      NOT NULL
);
GO

